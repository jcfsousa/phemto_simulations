# Project Name

A Python project that creats compton events at will. It outputs the data on a .t3pa data format

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Running the Project](#running-the-project)
- [About Virtual Environments](#about-virtual-environments)
---

## Installation

### Step 1: Clone the Repository
First, clone the project repository to your local machine:
```
git init 

git clone https://github.com/your-username/your-project.git
```

### Step 2: Install Python (if not already installed)

Make sure Python 3.6 or higher is installed. You can check your Python version by running:

```
python3 --version
```

If Python is not installed, download and install it from python.org.

### Step 3: Create a Virtual Environment

To keep the project dependencies isolated, it's recommended to use a virtual environment.
On Linux:

```
python3 -m venv venv
```

On Windows:

```
python -m venv venv
```

This will create a venv folder in your project directory containing the virtual environment.

### Step 4: Activate the Virtual Environment

After creating the virtual environment, activate it:

On Linux/macOS:

```
source venv/bin/activate
```

On Windows:

```
.\venv\Scripts\activate
```

Once activated, you should see (venv) in your terminal prompt, indicating that the virtual environment is active.

### Step 5: Install Project Dependencies

With the virtual environment active, install the project dependencies listed in the requirements.txt file:

```
pip install -e .
```

This will install all the required packages, including:
    numpy
    matplotlib
    pandas
    tqdm
    pyqt6

If you don’t have a requirements.txt file, you can manually install the packages like this:

```
pip install numpy matplotlib pandas tqdm pyqt6
```

### Step 6: Deactivate the Virtual Environment (Optional)

When you're done working in the virtual environment, you can deactivate it by running:

```
deactivate
```

## Usage

After installing the dependencies and activating the virtual environment, you can run the Python script using:

```
python3 compton_generator.py
```

## About Virtual Environments

Install venv on you machine

```
sudo apt install python3-venv
```

Creat a virtual environment. I typically creat a virtual environment on the father directory of the project directory. 
Use the following command:

```
python3 -m venv *name of environment*
```

To activate the environment run the following command:

```
source *name of the environment*/bin/activate
```

You can then work on the project. Within the environment you can pip install *package name* to add a certain package to the environment.

To deactivate the environment just run:

```
deactivate
```
