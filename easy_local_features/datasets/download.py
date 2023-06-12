import os, tqdm
# TODO: add output path

def retrieve(dataset="hpatches"):

    datasets_links = {
            "Alamo":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Alamo.tar", ## VALIDATION
            "EllisIsland":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Ellis_Island.tar", ## TEST
            "MadridMetropolis":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Madrid_Metropolis.tar",  ## ALL BELOW TRAIN
            "MontrealNotreDame":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Montreal_Notre_Dame.tar",
            "NYC_Library":"http://landmark.cs.cornell.edu/projects/1dsfm/images.NYC_Library.tar",
            "PiazzadelPopolo":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Piazza_del_Popolo.tar",
            "Piccadilly":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Piccadilly.tar",
            "RomanForum":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Roman_Forum.tar",
            "TowerofLondon":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Tower_of_London.tar",
            "Trafalgar":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Trafalgar.tar",
            "UnionSquare":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Union_Square.tar",
            "ViennaCathedral":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Vienna_Cathedral.tar",
            "Yorkminster":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Yorkminster.tar",
            "Gendarmenmarkt":"http://landmark.cs.cornell.edu/projects/1dsfm/images.Gendarmenmarkt.tar",
            }

    os.makedirs('downloads', exist_ok=True)

    # download hpatches http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
    
    if dataset == "hpatches":
        print("Downloading hpatches")
        if os.system("wget -nc http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz -O downloads/hpatches-sequences-release.tar.gz") or \
        os.system("tar -xv --skip-old-files -f downloads/hpatches-sequences-release.tar.gz -C downloads/"):
            raise RuntimeError('Failed to parse dataset.')

        # os.system("rm downloads/hpatches-sequences-release.tar.gz") ### If you want to save space uncomment this line to remove the tar files

    elif dataset == "sfm":
        os.makedirs("downloads/sfm", exist_ok=True)
        for key, item in tqdm.tqdm( datasets_links.items() ):
            if not os.path.isdir("downloads/sfm/" + key):
                print("Downloading " + key )
                if os.system("wget -nc "+ item + " -O downloads/sfm" + key+".tar") or \
                os.system("tar -xv --skip-old-files -f downloads/sfm" + key+".tar -C downloads/sfm" ):
                    raise RuntimeError('Failed to parse dataset.')

                os.system("rm downloads/sfm/" + key+".tar") ### If you want to save space uncomment this line to remove the tar files

    else:
        raise RuntimeError('Dataset not supported.')
    
def sfm():
    retrieve("sfm")

def hpatches():
    retrieve("hpatches")
