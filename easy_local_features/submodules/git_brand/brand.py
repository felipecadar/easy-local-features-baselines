import os, sys

class BRAND_CPP:
    def __init__(self):
        # this file path
        self.compile_path = os.path.dirname(os.path.realpath(__file__))

    def compile(self):
        # check if has cmake
        if os.system('cmake --version') != 0:
            print('cmake not found')
            sys.exit(1)

        # compile
        # rm build
        os.system('rm -rf ' + self.compile_path + '/build')
        # mkdir build
        os.system('mkdir -p ' + self.compile_path + '/build')
        # cd build
        os.chdir(self.compile_path + '/build')
        # cmake ..
        os.system('cmake ..')
        # make
        os.system('make')
        # cd ..
        os.chdir(self.compile_path)

if __name__ == "__main__":
    
    brand = BRAND_CPP()
    brand.compile()