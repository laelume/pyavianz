
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean
# Updated June 2026 laelume aka Ashlae Blum(e)

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Core batch processing

import os, re
import time
import soundfile as sf

# replaced 'from src.core import ...'

from . import spectrogram
from . import annotation
from . import config_loader
from . import batch_log
from . import bird_detector
from . import bat_detector
from . import segmentation
import logging

# Logging

VERBOSE = True    # toggle verbose debug logging on/off for this module

logger = logging.getLogger("batch_processor")
logger.setLevel(logging.DEBUG if VERBOSE else logging.WARNING)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)

# Constants
SAMPLES_PER_PAGE_16KHZ = 900 * 16000
MIN_FILE_SIZE_BYTES = 1000
SAMPLES_PER_300S_PAGE = 300

# Batch annotations never go next to the source audio, so a run can never
# overwrite <audio>.data (which may hold manual or reviewed annotations).
# Without a custom directory they go to a sibling of the analysed folder:
#   <analysed folder> + DEFAULT_ANNOTATION_DIR_SUFFIX
DEFAULT_ANNOTATION_DIR_SUFFIX = "_annotations"
# Scratch outputs of recogniser testing (training.py reads them from beside the audio)
TESTMODE_SUFFIXES = (".tmpdata", ".tmp2data")

class BatchProcessorCallbacks:
    """Abstract interface for user interaction callbacks - allows different implementations for CLI vs GUI"""
    
    def ask_resume_analysis(self, message):
        """Ask user if they want to resume previous analysis. Returns True to resume."""
        raise NotImplementedError
        
    def confirm_analysis_launch(self, message):
        """Ask user to confirm analysis parameters. Returns True to proceed."""
        raise NotImplementedError
        
    def update_progress(self, current, total, message):
        """Update progress display"""
        pass
        
    def check_cancelled(self):
        """Check if user has requested cancellation. Returns True if cancelled."""
        return False

class BatchProcessor:
    """Core batch processor for automated species detection across multiple audio files"""

    def __init__(self, configdir='', directory='', recognisers=None, 
                 callbacks=None,
                 subset=False, intermittent=False, 
                 wind="None", mergeSyllables=False, 
                 overwrite=None, overwriteSpecies=True, overwriteAll=False,
                 timeWindow_s=0, timeWindow_e=0,
                 protocolSize=15, protocolInterval=300, 
                 maxgap=1, minlen=0.2, maxlen=10,
                 annotationSaveDir=None, useGpu=True,
                 testmode=False):

        """Added support for annotaiton directory saving and gpu usage

        annotationSaveDir: directory to save annotation .data files into, mirroring the
                            source folder tree. If None, defaults to a sibling of the
                            analysed folder (<directory>_annotations). Annotations are
                            never saved alongside the source audio.
        useGpu:             whether GPU usage is permitted for NN inference, passed through
                            to configure_gpu_memory when NNmodel is constructed. If False,
                            forces cpu device without attempting cuda or prompting.
        """


        # Configuration
        self.configdir = configdir
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.ConfigLoader = config_loader.ConfigLoader()
        self.config = self.ConfigLoader.config(self.configfile)
        
        self.filtersDir = os.path.join(configdir, self.config['FiltersDir'])
        self.FilterDicts = self.ConfigLoader.filters(self.filtersDir)
        
        # Parameters
        self.dirName = directory
        self.annotationSaveDir = annotationSaveDir
        self.useGpu = useGpu

        # Backward compatibility: if overwrite is specified, use it for overwriteSpecies
        if overwrite is not None:
            self.overwriteSpecies = overwrite
            self.overwriteAll = False
        else:
            self.overwriteSpecies = overwriteSpecies
            self.overwriteAll = overwriteAll
        
        self.callbacks = callbacks
        self.testmode = testmode
        
        # Processing options stored as dictionary with named keys
        self.options = {
            'wind': wind,
            'subset': subset,
            'timeWindow_s': timeWindow_s,
            'timeWindow_e': timeWindow_e,
            'intermittent': intermittent,
            'protocolSize': protocolSize,
            'protocolInterval': protocolInterval,
            'mergeSyllables': mergeSyllables,
            'maxgap': maxgap,
            'minlen': minlen,
            'maxlen': maxlen
        }

        # Process species list
        if isinstance(recognisers, list):
            self.species = recognisers.copy()
        else:
            self.species = [recognisers]

        self.anySound = False
        if "Any sound" in self.species:
            self.anySound = True
            self.species.remove("Any sound")

        # Initialize detector classes
        self.bird_detector = bird_detector.BirdDetector(self.config, self.configdir)
        self.bat_detector = bat_detector.BatDetector()



    def has_annotations(self, filepath):
        """Checks if annotations may exist for this file, use custom annotation directory if one was set.

        Checks the resolved annotation save location (via define_annotation_save_location)
        rather than only the default alongside-audio path, so this correctly reflects
        prior results when --annotation-save-dir is in use.
        """
        resolvedDataPath = self.define_annotation_save_location(filepath, suffix=".data")
        base, _ = os.path.splitext(filepath)

        possible = [
            resolvedDataPath,
            base + ".txt",
            base + ".csv",
            base + ".json",
            base + ".xml",
            base + ".annotations"
        ]

        return any(os.path.exists(p) for p in possible)



    def define_annotation_save_location(self, filepath, suffix=".data"):
        """Resolves the path where annotations for filepath are read from and saved to.

        filepath: path to the source audio file being annotated
        suffix:   file suffix to append to the resolved output path
                  (".data" by default, overridden to ".tmpdata"/".tmp2data"
                  in testmode via saveAnnotation)

        Annotation directory (self.annotationDir, always set by process_files):
        mirrors the wav tree structure under that directory. This is the only
        location used for .data files.

        Alongside source audio: DISABLED so batch runs cannot overwrite existing
        <audio>.data files. Kept below, commented out, for reference.

        Pure path computation; directories are created by saveAnnotation.
        """
        base = os.path.splitext(os.path.basename(filepath))[0]

        # Testmode scratch files stay beside the audio: training.py reads them
        # from there, and they are never .data files.
        if suffix in TESTMODE_SUFFIXES:
            return filepath + suffix

        # OPTION (disabled): save alongside source audio file using its full path
        # if not self.annotationDir:
        #     return filepath + suffix

        # OPTION (active): annotation directory mirroring the wav tree structure
        sourceDir = os.path.abspath(os.path.dirname(filepath))
        drive, pathNoDrive = os.path.splitdrive(sourceDir)
        # strip leading separator so os.path.join doesn't treat this as an
        # absolute path and discard self.annotationDir
        pathNoDrive = pathNoDrive.lstrip(os.sep).lstrip("/")

        outDir = os.path.join(self.annotationDir, pathNoDrive)

        return os.path.join(outDir, base + suffix)

    @staticmethod
    def is_alongside_audio(filepath, outPath):
        """True if outPath would sit in the same folder as the source audio."""
        return os.path.normcase(os.path.abspath(os.path.dirname(outPath))) == \
               os.path.normcase(os.path.abspath(os.path.dirname(filepath)))

    def process_files(self):
        """Main processing method. Returns 0 on success, 1 on error."""
        filters = [self.FilterDicts[name] for name in self.species]
        
        samplerate = set([filt["SampleRate"] for filt in filters])
        if len(samplerate) > 1:
            print("Multiple sample rates required: ", samplerate)
            print("Audio will be resampled as needed for each filter group")

        speciesStr = " & ".join(self.species)

        self.NNDicts = self.ConfigLoader.getNNmodels(self.FilterDicts, self.filtersDir, self.species)
        
        # Setup annotation directory: custom if given, else a sibling of the
        # analysed folder. Never alongside the source audio.
        if self.annotationSaveDir:
            self.annotationDir = self.annotationSaveDir
        else:
            self.annotationDir = os.path.normpath(os.path.abspath(str(self.dirName))) + DEFAULT_ANNOTATION_DIR_SUFFIX
        print(f"Annotations will be saved under: {self.annotationDir}")
        
        allsoundfiles = self.get_files_to_process()
        total = len(allsoundfiles)

        # Check to see if there are any existing annotations
        any_existing_annotations = any(
            self.has_annotations(f) for f in allsoundfiles
        )

        self.filesDone = []
        self.log = batch_log.Log(os.path.join(self.dirName, 'LastAnalysisLog.txt'), speciesStr, self.options)
        
        if self.log.possibleAppend:
            filesExistAndDone = self.log.getDoneFiles(allsoundfiles)
            message = f"Previous analysis found in this folder (analysed {len(filesExistAndDone)} out of {total} files in this folder).\nWould you like to resume that analysis?"
            
            if self.callbacks.ask_resume_analysis(message):
                self.filesDone = filesExistAndDone
            else:
                self.filesDone = []

        cnt = len(self.filesDone)
        # Format options for display
        opts_parts = []
        if self.options['wind'] != "None":
            opts_parts.append(f"Wind: {self.options['wind']}")
        if self.options['subset']:
            opts_parts.append(f"Subset: {self.options['timeWindow_s']}-{self.options['timeWindow_e']}")
        if self.options['intermittent']:
            opts_parts.append(f"Intermittent: {self.options['protocolSize']}s every {self.options['protocolInterval']}s")
        if self.options['mergeSyllables']:
            opts_parts.append(f"Merge syllables: gap={self.options['maxgap']}, min={self.options['minlen']}, max={self.options['maxlen']}")
        opts = ', '.join(opts_parts) if opts_parts else 'None'
        
        message = f"Species: {speciesStr}, options: {opts}.\nNumber of files to analyse: {total}, {cnt} done so far.\n"
        message += f"Log file stored in {self.dirName}/LastAnalysisLog.txt.\n"



        # Provides updates about whether there are existing annotations
        if self.overwriteAll:

            filesToActuallyProcess = total - len(self.filesDone)
            if any_existing_annotations and filesToActuallyProcess > 0:
                message += f"\nWarning: ALL previous annotations will be deleted in the {filesToActuallyProcess} file(s) not already marked done!\n"
            elif any_existing_annotations and filesToActuallyProcess == 0:
                message += "\nAll files already marked done in log; no files will be reprocessed or overwritten this run.\n"
            else:
                message += "\nNo existing annotations found (nothing will be overwritten).\n"

        elif self.overwriteSpecies:
            filesToActuallyProcess = total - len(self.filesDone)
            if any_existing_annotations and filesToActuallyProcess > 0:
                message += f"\nWarning: any previous annotations for the selected species will be deleted in the {filesToActuallyProcess} file(s) not already marked done!\n"
            elif any_existing_annotations and filesToActuallyProcess == 0:
                message += "\nAll files already marked done in log; no files will be reprocessed or overwritten this run.\n"
            else:
                message += "\nNo existing annotations for selected species found.\n"


        if any_existing_annotations and not (self.overwriteAll or self.overwriteSpecies):
            message += "\nExisting annotations detected. Run with overwrite enabled to replace them.\n"

        message = "Analysis will be launched with these options:\n" + message + "\nConfirm?"



        if not self.callbacks.confirm_analysis_launch(message):
            print("Analysis cancelled")
            return 1

        self.log.file = open(self.log.filepath, 'w') 

        # Restore other species' previously logged analyses, which open('w')
        # would otherwise silently discard
        self.log.reprintOld()
        
        self.log.appendHeader(header=None, species=self.log.species, settings=self.log.settings)


        # If resuming, re-write the previously completed files for this species
        # back into the log immediately, so progress from the prior run isn't
        # lost from disk if this run is interrupted before reaching new files
        for doneFile in self.filesDone:
            self.log.file.write(doneFile)
            self.log.file.write("\n")
        self.log.file.flush()

        self.callbacks.update_progress(cnt, total, "Preparing for analysis...")

        return self.process_file_loop(allsoundfiles, total, filters)

    def get_files_to_process(self):
        """Get list of all files that will be processed"""
        allsoundfiles = []
        
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                isBatMode = any("NZ Bats" in species or species == "NZ Bats_NP" for species in self.species)
                
                if not isBatMode and (filename.lower().endswith('.wav') or filename.lower().endswith('.flac')):
                    allsoundfiles.append(os.path.join(root, filename))
                elif isBatMode:
                    if filename.lower().endswith('.bmp'):
                        allsoundfiles.append(os.path.join(root, filename))
                    
        return allsoundfiles

    def process_file_loop(self, allsoundfiles, total, filters):
        """Main file processing loop"""
        processingTime = 0
        cnt = 0
        
        timeWindow_s = self.options['timeWindow_s']
        timeWindow_e = self.options['timeWindow_e']

        # Track remaining file counts per directory, so a directory-complete
        # marker can be logged as soon as every file in it has been handled,
        # whether processed successfully or skipped (already done, invalid,
        # or outside the time window)
        remainingPerDir = {}
        for f in allsoundfiles:
            d = os.path.dirname(f)
            remainingPerDir[d] = remainingPerDir.get(d, 0) + 1
        for doneFile in self.filesDone:
            d = os.path.dirname(doneFile)
            if d in remainingPerDir:
                remainingPerDir[d] -= 1

        # new function to track progress and completion
        def markHandled(filepath):
            """ Decrements the remaining-file count for filepath's directory, and logs a directory-complete marker once that count reaches zero. """
            fileDir = os.path.dirname(filepath)
            if fileDir in remainingPerDir:
                remainingPerDir[fileDir] -= 1
                if remainingPerDir[fileDir] <= 0:
                    self.log.appendDirectoryComplete(fileDir)

        for filename in allsoundfiles:
            if self.callbacks.check_cancelled():
                print("Processing cancelled by user")
                return 1
                
            processingTimeStart = time.time()
            hh, mm = divmod(processingTime * (total-cnt) / 60, 60)
            cnt = cnt + 1
            progrtext = f"file {cnt} / {total}. Time remaining: {int(hh)} h {mm:.2f} min"
            
            self.callbacks.update_progress(cnt, total, progrtext)
            print(f"*** Processing {progrtext} ***")

            # Skip if already processed
            if filename in self.filesDone:
                print(f"File {filename} processed previously, skipping")
                continue

            # Validate file
            if not self.validate_file(filename):
                markHandled(filename)
                continue

            # Check time window for DOC recordings
            if not self.check_time_window(filename, timeWindow_s, timeWindow_e):
                markHandled(filename)
                continue

            success = self.process_single_file(filename, filters)
            if success:
                # self.log.appendFile(filename)
                self.log.appendFile(filename, annotationsFound=self.lastAnnotationsFound)
                markHandled(filename)


            processingTime = time.time() - processingTimeStart
            print(f"File processed in {processingTime}")

        print(f"Processed all {total} files")
        return 0

    def validate_file(self, filename):
        """Validate that file exists, has content, and is properly formatted"""
        if os.stat(filename).st_size < MIN_FILE_SIZE_BYTES:
            print(f"File {filename} empty, skipping")
            return False

        isBatMode = any("NZ Bats" in species or species == "NZ Bats_NP" for species in self.species)
        
        with open(filename, 'br') as f:
            first2char = f.read(2)
            f.seek(0)
            first4char = f.read(4)
            
            isValidFormat = False
            
            if isBatMode:
                isValidFormat = (first2char == b'BM') or (first4char == b'RIFF') or (first4char == b'fLaC')
            else:
                isValidFormat = (first4char == b'RIFF') or (first4char == b'fLaC')
            
            if not isValidFormat:
                print(f"File {filename} is not a valid audio/BMP file, skipping")
                return False

        return True

    def check_time_window(self, filename, timeWindow_s, timeWindow_e):
        """Check if DOC recording falls within specified time window"""
        DOCRecording = re.search(r'(\d{6})_(\d{6})', os.path.basename(filename))
        if not DOCRecording:
            return True
            
        startTime = DOCRecording.group(2)
        sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
        
        if timeWindow_s == timeWindow_e:
            inWindow = True
        elif timeWindow_s < timeWindow_e:
            inWindow = timeWindow_s <= sTime <= timeWindow_e
        else:
            inWindow = sTime >= timeWindow_s or sTime <= timeWindow_e
            
        if not inWindow:
            print(f"Skipping out-of-time-window recording {filename}")
            
        return inWindow


    def process_single_file(self, filename, filters):
        """Process a single file. Returns True on success."""
        print("Loading file...")
        self.currentFilename = filename
        
        isBatMode = any("NZ Bats" in species or species == "NZ Bats_NP" for species in self.species)
        
        self.loadFile(filename, isBatMode)
        
        print('Segments in this file: ', self.segments)
        startCount = len(self.segments)

        # Initialize segments_nonn for testmode
        if self.testmode:
            self.segments_nonn = annotation.SegmentList()

        if self.options['intermittent']:
            self.addRegularSegments(filename, self.options['protocolSize'], self.options['protocolInterval'])
        else:
            self.detectFile(filters)

        newSegmentCount = len(self.segments) - startCount
        print(f"{newSegmentCount} new segments marked")
        
        # Save annotations
        if self.testmode:
            # Save separately with and without NN
            self.saveAnnotation(filename, self.segments, suffix=".tmpdata")
            self.saveAnnotation(filename, self.segments_nonn, suffix=".tmp2data")
        else:
            self.saveAnnotation(filename, self.segments)
        
        # annotationsFound reflects total segments present after processing,
        # not just newly-added ones, since a resumed/appended file could
        # already have prior segments loaded in loadFile
        self.lastAnnotationsFound = len(self.segments) > 0        

        return True




    def detectFile(self, filters):
        """Actual worker for a file in the detection loop."""
        # Check if this is bat processing
        if any('NZ Bats' in species or species == "NZ Bats_NP" for species in self.species):
            self.bat_detector.detectBatsInFile(
                sp=self.sp,
                segments=self.segments,
                currentFilename=self.currentFilename,
                filters=filters,
                NNDicts=self.NNDicts,
                testmode=self.testmode,
                check_cancelled=self.callbacks.check_cancelled
            )
        else:
            # Pass segments_nonn only in testmode
            segments_nonn = self.segments_nonn if self.testmode else None
            
            self.bird_detector.detectBirdsInFile(
                sp=self.sp,
                segments=self.segments,
                species=self.species,
                filters=filters,
                NNDicts=self.NNDicts,
                options=self.options,
                anySound=self.anySound,
                testmode=self.testmode,
                segments_nonn=segments_nonn,
                check_cancelled=self.callbacks.check_cancelled, 
                useGpu=self.useGpu
            )

    def loadFile(self, filename, bats=False, anysound=False, impMask=False):
        """Load audio file and prepare for processing."""
        self.sp = spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])

        if bats:
            self.sp.readSoundFile(filename, rotate=False)
        else:
            self.sp.readSoundFile(filename)

        if self.sp.audio_data.data is not None:
            print("Read %d samples, %f s at %d Hz" % (len(self.sp.audio_data.data), float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate, self.sp.audio_data.sample_rate))
        else:
            duration = self.sp.get_duration()
            print("Read BMP spectrogram: %d x %d pixels, %f s at %d Hz" % (self.sp.sg.shape[0], self.sp.sg.shape[1], duration, self.sp.audio_data.sample_rate))

        self.segments = annotation.SegmentList()
        
        duration = self.sp.get_duration()


        # Resolve the actual annotation path, honouring annotationDir if set,
        # rather than assuming the default alongside-audio location. Without
        # this, a custom --annotation-save-dir would never be checked here,
        # and prior results saved there would be silently overwritten with
        # no overwriteSpecies/overwriteAll logic ever applying to them.
        existingAnnotationPath = self.define_annotation_save_location(filename, suffix=".data")


        # # If overwriteAll is set, or if we're in bat/anysound mode, or no .data file exists:
        # # wipe everything
        # if self.overwriteAll or bats or anysound or not os.path.isfile(filename + '.data'):
        
        # If overwriteAll is set, or if we're in bat/anysound mode, or no existing
        # annotation file exists at the resolved location: wipe everything
        if self.overwriteAll or bats or anysound or not os.path.isfile(existingAnnotationPath):
            self.segments.metadata["Operator"] = "Auto"
            self.segments.metadata["Reviewer"] = ""
            self.segments.metadata["Duration"] = duration
            print("Wiping all previous segments")
            self.segments.clear()
        else:
            # Load existing annotations
            # hasmetadata = self.segments.parseJSON(filename+'.data', duration)
            hasmetadata = self.segments.parseJSON(existingAnnotationPath, duration)
            if not hasmetadata:
                self.segments.metadata["Operator"] = "Auto"
                self.segments.metadata["Reviewer"] = ""
                self.segments.metadata["Duration"] = duration
            
            # If overwriteSpecies is set, remove annotations for the selected species
            if self.overwriteSpecies:
                for species in self.species:
                    if species in self.FilterDicts:
                        spname = self.FilterDicts[species]["species"]
                        print("Wiping species", spname)
                        oldsegs = self.segments.getSpecies(spname)
                        for i in reversed(oldsegs):
                            wipeAll = self.segments[i].wipeSpecies(spname)
                            if wipeAll:
                                del self.segments[i]
            # If neither overwrite option is set, keep all existing annotations
            # and just add new detections
            print("%d segments loaded from .data file" % len(self.segments))

    def saveAnnotation(self, filename, segmentList, suffix=".data"):
        """Generates default batch-mode metadata and saves segmentList to the annotation directory.

        Resolves the output path via define_annotation_save_location. Refuses to
        write a .data file into the source audio's folder, so existing
        <audio>.data annotations can never be overwritten by a batch run.
        """
        segmentList.metadata["Operator"] = "Auto"
        segmentList.metadata["Reviewer"] = ""
        segmentList.metadata["Duration"] = self.sp.get_duration()
        segmentList.metadata["noiseLevel"] = None
        segmentList.metadata["noiseTypes"] = []

        # OPTION (disabled): write annotation to wav source location ("safe backup")
        # segmentList.saveJSON(str(filename) + suffix)

        # write annotation to the annotation directory
        outPath = self.define_annotation_save_location(filename, suffix=suffix)
        if suffix not in TESTMODE_SUFFIXES and self.is_alongside_audio(filename, outPath):
            raise RuntimeError(
                f"Refusing to save {outPath}: annotations must not be written next to the "
                f"source audio. Choose an annotation directory outside the audio folders."
            )
        os.makedirs(os.path.dirname(outPath), exist_ok=True)
        logger.debug("Saving annotation for %s to %s", filename, outPath)
        segmentList.saveJSON(outPath)

        return 1


    def addRegularSegments(self, filename, length, interval):
        """Perform the Hartley bodge: add fixed length segments at specified interval."""
        info = sf.info(filename)
        samplerate = info.samplerate
        nseconds = info.frames / samplerate
        self.segments.metadata["Operator"] = "Auto"
        self.segments.metadata["Reviewer"] = ""
        self.segments.metadata["Duration"] = nseconds
        i = 0
        segments = []
        print("Adding segments (%d s every %d s) to %s" %(length,interval, str(filename)))
        while i < nseconds:
            end_time = min(i + length, nseconds)
            segments.append([i, end_time])
            i += interval
        post = segmentation.PostProcess(configdir=self.configdir, audioData=None, sampleRate=0, segments=segments, subfilter={}, cert=0)
        self.segments.addFromTimeRanges(post.segments, 0, 0, species="Don't Know", certainty=0.0)