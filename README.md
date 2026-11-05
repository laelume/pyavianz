# **AVIANZ** : an open-source biacoustics analysis software

0>0

## Welcome to the Pythonically packaged version of Avianz
This fork lets you interact directly with Avianz via pythonic API-like calls to make your life easier, or maybe harder. 

## Launch avianz with a single keyword from the terminal (requires package install)
```
cd avianz
pip install -e .
avianz
```

Features: 
* review and listen to wav files, 
* segment and annotate recordings, 
* train wavelet filters to recognise species, 
* import custom filters and batch process files,
* review annotations and export as csv or JSON, 

For more information, see http://www.avianz.net

# Environment

Supports Python <= 3.10

## Using conda
```
conda env create -n avianz -f environment.yml -y python==3.10
```

## Editable python package
```
git clone https://github.com/laelume/avianz.git && cd avianz && pip install -e .
```

# CLI Command Line Interface
```
cd avianz && 
python AviaNZ.py -c -b -d "Z:\some_top_level\collection\of\audio_files\YYYYMMDD" -r "Kiwi (Nth Is Brown)" -w -asd "D:\avianz_annotations"
```

Batch mode never writes `.data` files next to the source audio, so existing annotations there are never overwritten. Results go to the `-asd` directory, mirroring the audio folder tree:
`-asd D:\out` + `Z:\survey\site1\rec.wav` → `D:\out\survey\site1\rec.data`.
Without `-asd`, they go to `<input dir>_annotations` beside the input folder, which must be writable. Existing `rec.wav.data` files beside the audio are not read or merged by batch runs. Note: the GUI review windows still only open `<audio>.data` beside the audio, so they don't yet read batch results from the annotation directory.

# GUI Installation

## Windows
Windows binaries are available at http://www.avianz.net.
To install from source, follow the Linux instructions.

- Remember to include a C compiler such as VS Build Tools, and make sure to check "Desktop development with C++"

(PS)
```
winget install Microsoft.VisualStudio.2022.BuildTools
``

OR POSSIBLY:
```winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools"
```

(then restart PC)

## macOS
An installer script is available at http://www.avianz.net.
To install from source, follow the Linux instructions.

## Linux
No binaries are available. Install from the source as follows:
1. Download the source .zip of the latest release.
2. Extract (`unzip v2.0.zip`) and navigate to the extracted directory.
3. Ensure Python (3.6 or higher), pip and git are available on your system. On Ubuntu, these can be installed by running:  
```
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo apt-get install git
```
4. Install the required packages by running `pip3 install -r requirements.txt --user` at the command line. (On Ubuntu and some other systems, `python` and `pip` refer to the Python 2 versions. If you are sure these refer to version 3 of the language, use `python` and `pip` in steps 4-6.)  
5. Build the Cython extensions by running `cd ext; python3 setup.py build_ext -i; cd..`  
6. Done! Launch the software with `python3 AviaNZ.py`  


# Acknowledgements
This fork is maintained by Ashlae Blum'e. 

Original software was developed by Stephen Marsland et alia (see below).  

AviaNZ is based on PyQtGraph and PyQt, and uses Librosa and Scikit-learn amongst others.

Development of this software was supported by the RSNZ Marsden Fund, and the NZ Department of Conservation.

# Citation

If you use this software, please credit us in any papers that you write. An appropriate reference is:

```
@article{Marsland19,
  title = "AviaNZ: A future-proofed program for annotation and recognition of animal sounds in long-time field recordings",
  author = "{Marsland}, Stephen and {Priyadarshani}, Nirosha and {Juodakis}, Julius and {Castro}, Isabel",
  journal = "Methods in Ecology and Evolution",
  volume = 10,
  number = 8,
  pages = "1189--1195",
  year = 2019, 
  url = "http://www.avianz.net"
  doi = "10.5061/dryad.m70p89d"
}
```
