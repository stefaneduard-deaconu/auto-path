echo "The old ./venv/ folder will be removed"
rm -r venv # remove possible older venv
python3.10 -m venv venv
source venv/bin/activate
pip3.10 install numpy \
                scipy \
                matplotlib \
                redis \
                celery
pip3.10 install git+https://github.com/pvigier/perlin-numpy
