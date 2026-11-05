
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean
# Updated June 2026 laelume aka Ashlae Blum

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

# This is the script that starts AviaNZ. It processes command line options
# and then calls either part of the GUI, or runs on the command line directly.

import click

# Command line interface examples for species filters

# For inference
# python AviaNZ.py -c -b -d "/source/audio/directory" -r "Morepork" -w

# For training
# python AviaNZ.py -c -t -d "/source/audio/directory" -e "/secondary/source/audio/directory" -r "Morepork" -x 2

# For testing
# python AviaNZ.py -c -u -d "/source/audio/directory" -r "Kiwi (Tokoeka Rakiura)"

# Generate images without GUI
# python AviaNZ.py -c -s -f "image_save_location"

@click.command()
@click.option('-c', '--cli', is_flag=True, help='Run in command-line mode')
@click.option('-s', '--cheatsheet', is_flag=True, help='Make the cheatsheet images')
@click.option('-z', '--zooniverse', is_flag=True, help='Make the Zooniverse images and sounds')
@click.option('-f', '--infile', type=click.Path(), help='Input wav file (mandatory directory in CLI mode)')
@click.option('-o', '--imagefile', type=click.Path(), help='If specified, a spectrogram will be saved to this file')
@click.option('-b', '--batchmode', is_flag=True, help='Batch processing')
@click.option('-t', '--training', is_flag=True, help='Train a NN recogniser')
@click.option('-u', '--testing', is_flag=True, help='Train a recogniser')
@click.option('-d', '--sdir1', type=click.Path(), help='Input sound directory, training or batch processing')
@click.option('-e', '--sdir2', type=click.Path(), help='Second input sound directory, training')

# @click.option('-r', '--recogniser', type=str, help='Recogniser name (without ".txt"), batch processing')
@click.option('-r', '--recogniser', type=str, multiple=True, help='Recogniser name(s) (without ".txt"), batch processing. Repeat -r for multiple species, e.g. -r "Kiwi (Nth Is Brown)" -r "Kiwi (Little Spotted)"')

@click.option('-w', '--wind', is_flag=True, help='Apply wind filter')
@click.option('-x', '--width', type=float, help='Width of windows for NN')
@click.option('--time-start', type=int, default=0, help='Start time for subset (seconds from midnight, 0-86400)')
@click.option('--time-end', type=int, default=0, help='End time for subset (seconds from midnight, 0-86400)')
@click.option('--protocol-size', type=int, default=15, help='Length of segments for intermittent sampling (seconds)')
@click.option('--protocol-interval', type=int, default=300, help='Interval between segments for intermittent sampling (seconds)')
@click.option('--maxgap', type=float, default=1.0, help='Maximum gap between syllables to merge (seconds)')
@click.option('--minlen', type=float, default=0.2, help='Minimum syllable length (seconds)')
@click.option('--maxlen', type=float, default=10.0, help='Maximum syllable length (seconds)')
@click.option('--subset/--no-subset', default=False, help='Enable time-limited subset processing')
@click.option('--intermittent/--no-intermittent', default=False, help='Enable intermittent sampling')
@click.option('--merge-syllables/--no-merge-syllables', default=False, help='Enable syllable merging')
# Annotation save location: batch mode no longer saves alongside source audio, so the
# default changed from "beside each .wav" to <input dir>_annotations (see batch_processor.py)
# @click.option('-asd', '--annotation-save-dir', type=click.Path(), default=None, help='Custom directory to save annotation .data files into, instead of alongside source audio')
@click.option('-asd', '--annotation-save-dir', type=click.Path(), default=None, help='Directory to save annotation .data files into, mirroring the audio folder tree (default: <input dir>_annotations). Batch mode never saves alongside source audio')
@click.option('--gpu/--no-gpu', default=True, help='Allow GPU usage for NN inference if available; prompts for CPU fallback confirmation if not')
@click.argument('command', nargs=-1)

def mainlauncher(cli, cheatsheet, zooniverse, infile, imagefile, batchmode, training, testing, sdir1, sdir2, recogniser, wind, width, time_start, time_end, protocol_size, protocol_interval, maxgap, minlen, maxlen, subset, intermittent, merge_syllables, annotation_save_dir, gpu, command):
    # Suppress TensorFlow messages (if it gets imported somehow)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # adapt path to allow this to be launched from wherever
    import sys
    if getattr(sys, 'frozen', False):
        appdir = sys._MEIPASS
    else:
        appdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(appdir)

    #print("Using python at", sys.path)
    #print(os.environ)
    #print("Version", sys.version)

    try:
        import platform, json, shutil
        from jsonschema import validate
        from avianz.src.core import config_loader
    except Exception as e:
        print("ERROR: could not import packages")
        raise

    # determine location of config file and bird lists
    if platform.system() == 'Windows':
        # Win
        configdir = os.path.expandvars(os.path.join("%APPDATA%", "AviaNZ"))
    elif platform.system() == 'Linux' or platform.system() == 'Darwin':
        # Unix
        configdir = os.path.expanduser("~/.avianz/")
    else:
        print("ERROR: what OS is this? %s" % platform.system())
        raise
    #print(configdir)

    # if config and bird files not found, copy from distributed backups.
    # so these files will always exist on load (although they could be corrupt)
    # (exceptions here not handled and should always result in crashes)
    if not os.path.isdir(configdir):
        print("Creating config dir %s" % configdir)
        try:
            os.makedirs(configdir)
        except Exception as e:
            print("ERROR: failed to make config dir")
            print(e)
            raise

    # pre-run check of config file validity
    confloader = config_loader.ConfigLoader()
    configschema = json.load(open("Config/config.schema"))
    learnparschema = json.load(open("Config/learnpar.schema"))
    try:
        config = confloader.config(os.path.join(configdir, "AviaNZconfig.txt"))
        validate(instance=config, schema=configschema)
        learnpar = confloader.learningParams(os.path.join(configdir, "LearningParams.txt"))
        validate(instance=learnpar, schema=learnparschema)
        print("successfully validated config file")
    except Exception as e:
        # NOTE: Gives a QWidget error instead of this
        print("Warning: config file failed validation with:")
        print(e)
        try:
            shutil.copy2("Config/AviaNZconfig.txt", configdir)
            shutil.copy2("Config/LearningParams.txt", configdir)
        except Exception as e:
            print("ERROR: failed to copy essential config files")
            print(e)
            raise
        # reload the restored default: `config` is unset if loading failed (missing or corrupt
        # file, e.g. first run) and stale if validation failed, but GUI mode reads previousMode/File from it
        config = confloader.config(os.path.join(configdir, "AviaNZconfig.txt"))

    # check and if needed copy any other necessary files
    necessaryFiles = ["ListCommonBirds.txt", "ListDOCBirds.txt", "ListBats.txt", "LearningParams.txt", "ListKnownCalls.txt", "Freebird_species_list.xlsx"]
    for f in necessaryFiles:
        if not os.path.isfile(os.path.join(configdir, f)):
            print("File %s not found in config dir, providing default" % f)
            try:
                shutil.copy2(os.path.join("Config", f), configdir)
            except Exception as e:
                print("ERROR: failed to copy essential config files")
                print(e)
                raise

    # copy over filters to ~/.avianz/Filters/:

    # Walk the recogniser tree.
    filterdir = os.path.join(configdir, "Filters/")
    if not os.path.isdir(filterdir):
        print("Creating filter dir %s" % filterdir)
        os.makedirs(filterdir)

    for root, dirs, files in os.walk("Filters"):

        # Ignore Python bytecode cache directories.
        if "__pycache__" in dirs:
            print("Ignoring __pycache__ directory.")
            dirs.remove("__pycache__")

        for f in files:
            ff = os.path.join(root, f) # Kiwi.txt

            # Preserve the relative directory structure under Filters/
            rel = os.path.relpath(ff, "Filters")
            dest = os.path.join(filterdir, rel) # ~/.avianz/Filters/Kiwi.txt

            # Create destination subdirectories if needed.
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            if not os.path.isfile(dest): # ~/.avianz/Filters/Kiwi.txt
                print("Recogniser %s not found, providing default" % rel)
                try:
                    shutil.copy2(ff, dest) # cp Filters/Kiwi.txt ~/.avianz/Filters/
                except Exception as e:
                    print("Warning: failed to copy recogniser %s to %s" % (ff, dest))
                    print(e)


    # run splash screen:
    if cli:
        print("Starting AviaNZ in CLI mode")
        if batchmode:
            from src.cli.batch_cli import run_cli_batch
            # if os.path.isdir(sdir1) and recogniser in confloader.filters(filterdir).keys():
            # Added multi-species filter inclusion
            # recogniser is a tuple; an empty one (no -r given) must fail, as all() of nothing is True
            if os.path.isdir(sdir1) and recogniser and all(r in confloader.filters(filterdir).keys() for r in recogniser):

                wind_str = "OLS wind filter (recommended)" if wind else "None"

                # Confirms annotation directory (batch annotations never go alongside source audio)
                if annotation_save_dir:
                    print(f"Annotations will be saved to custom directory: {annotation_save_dir}")
                else:
                    # OPTION (disabled): save alongside source audio files
                    # print("Annotations will be saved alongside source audio files (default)")
                    print(f"Annotations will be saved to default directory: {os.path.normpath(os.path.abspath(sdir1))}_annotations")

                # Confirms user-selected GPU setting
                print(f"GPU usage: {'enabled (falls back to CPU prompt if unavailable)' if gpu else 'disabled, forcing CPU'}")

                result = run_cli_batch(
                    configdir=configdir, 
                    directory=sdir1, 
                    recognisers=list(recogniser), 
                    subset=subset, 
                    intermittent=intermittent, 
                    wind=wind_str, 
                    mergeSyllables=merge_syllables, 
                    overwrite=True, 
                    timeWindow_s=time_start, 
                    timeWindow_e=time_end, 
                    protocolSize=protocol_size, 
                    protocolInterval=protocol_interval, 
                    maxgap=maxgap, 
                    minlen=minlen, 
                    maxlen=maxlen,
                    annotationSaveDir=annotation_save_dir,
                    useGpu=gpu                    
                )
                if result == 0:
                    print("Analysis complete, closing AviaNZ")
                else:
                    print("Analysis failed")
                    sys.exit(result)
            else:
                print("ERROR: valid input dir (-d) and recogniser name (-r) are essential for batch processing")
                raise
        elif training:
            from avianz.src.core import training
            # -r is multiple=True (a tuple); training takes exactly one recogniser
            rec = recogniser[0] if len(recogniser) == 1 else None
            if os.path.isdir(sdir1) and os.path.isdir(sdir2) and rec in confloader.filters(filterdir).keys() and width>0:
                training = training.NNTrain(configdir,filterdir,sdir1,sdir2,rec,width,CLI=True)
                training.cliTrain()
                print("Training complete, closing AviaNZ")
            else:
                print("ERROR: valid input dirs (-d and -e) and exactly one recogniser name (-r) are essential for training")
                raise
        elif testing:
            from avianz.src.core import training
            filts = confloader.filters(filterdir)
            # -r is multiple=True (a tuple); testing takes exactly one recogniser
            rec = recogniser[0] if len(recogniser) == 1 else None
            if os.path.isdir(sdir1) and rec in filts:
                testing = training.NNTest(sdir1, filts[rec], rec, configdir,filterdir,CLI=True)
                print("Testing complete, closing AviaNZ")
            else:
                print("ERROR: valid input dir (-d) and exactly one recogniser name (-r) are essential for testing")
                raise
        else:
            if (cheatsheet or zooniverse) and isinstance(infile, str):
                from PyQt6.QtWidgets import QApplication
                from src.ui import AviaNZ_manual_GUI
                #from src.ui import manual_interface
                app = QApplication(sys.argv)
                avianz = AviaNZ_manual_GUI.AviaNZ(configdir=configdir, CLI=True, cheatsheet=cheatsheet, zooniverse=zooniverse, firstFile=infile, imageFile=imagefile, command=command)
                #avianz = manual_interface.ManualInterface(configdir=configdir, CLI=True, cheatsheet=cheatsheet, zooniverse=zooniverse, firstFile=infile, imageFile=imagefile, command=command)
                print("Analysis complete, closing AviaNZ")
            else:
                print("ERROR: valid input file (-f) is needed")
                raise
    else:
        # task = None # removed because we want to load from preset config state to speed up the process. 
        print("Starting AviaNZ in GUI mode")
        from PyQt6.QtWidgets import QApplication
        from PyQt6 import QtCore
        
        # Register the UI MessagePopup implementation for core modules to use
        from src.ui.components.popups import MessagePopup as UIMessagePopup
        from avianz.src.core import message_popup
        message_popup.set_message_popup_class(UIMessagePopup)
        
        #QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication(sys.argv)
        # a hack to fix default font size (Win 10 suggests 7 pt for QLabels for some reason)
        QApplication.setFont(QApplication.font("QMenu"))

        # (つ -' _ '- )つ

        # read previous session state from config to restore task and audio file on relaunch
        _prev_audio_file = config.get("previousFile", "")
        _prev_task = config.get("previousMode", None)
        if _prev_task not in (1, 2, 3, 4):
            _prev_task = None
        print("Previous session: task=%s, file=%s" % (_prev_task, _prev_audio_file))

        # seed task from previous session to skip splash screen on relaunch
        task = _prev_task

        # saves session state to AviaNZconfig.txt after dialogs complete, before main window loads
        def _save_preset(avianz_instance):
            _cfg = confloader.config(os.path.join(configdir, "AviaNZconfig.txt"))
            _cfg["previousMode"] = task
            _cfg["previousFile"] = avianz_instance.filename if hasattr(avianz_instance, "filename") and avianz_instance.filename else ""
            confloader.configwrite(_cfg, os.path.join(configdir, "AviaNZconfig.txt"))
            print("Session state saved to AviaNZconfig.txt")

        # (つ -' _ '- )つ
        
        while True:
            # Opens startup splash screen
            if task is None:
                # This screen asks what you want to do, then processes the response
                from src.ui.dialogs.start_screen import StartScreen
                first = StartScreen()
                first.show()
                app.exec()
                task = first.getValues()

            # Initialize to previous task for skip splash screen on relaunch
            avianz = None
            if task == 1:
                from src.ui import manual_interface
                # Loads config from file
                avianz = manual_interface.ManualInterface(configdir=configdir, firstFile=_prev_audio_file, on_ready=_save_preset)
                # avianz = manual_interface.ManualInterface(configdir=configdir)
            elif task==2:
                from src.ui import batch_interface
                avianz = batch_interface.BatchInterface(configdir=configdir)
            elif task==3:
                from src.ui import review_interface
                avianz = review_interface.ReviewInterface(configdir=configdir)
            elif task==4:
                from src.ui import split_interface
                avianz = split_interface.SplitData()

            # catch bad initialiation
            if avianz:
                avianz.activateWindow()
            else:
                return

            out = app.exec()
            QApplication.closeAllWindows()
            QApplication.processEvents()

            # catch exit code to see if restart requested:
            # (note: do not use this for more complicated cleanup,
            #  no guarantees that it is returned before program closes)
            if out == 0:
                # default quit
                break
            elif out == 1:
                # restart to splash screen
                task = None
            elif out == 2:
                # request switch to Splitter
                task = 4


try:
    mainlauncher()
except Exception:
    import traceback
    print(traceback.format_exc())
    input("Encountered error. Report it with the text above to AviaNZ team at www.avianz.net.\nPress ENTER to exit")
    raise
