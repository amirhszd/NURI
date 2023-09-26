"""
    @author: Eon Rehman
"""

def get_header_file_reflectance_conv(Header_File,Output_Header_File):

    f = open(Header_File, 'r')
    linelist = f.readlines()
    f.close

    # Re-open file here
    f2 = open(Output_Header_File, 'w')
    for line in linelist:
        line = line.replace('{HEADWALL Hyperspec III[HEADWALL Hyperspec III],[HEADWALL Hyperspec III OR]}',
                            '{HEADWALL Hyperspec III[HEADWALL Hyperspec III],[HEADWALL Hyperspec III OR], [Eon, Refl]}')
        line = line.replace('data type = 12', 'data type = 4')
        line = line.replace('bsq','bip')
        f2.write(line)
    f2.close()

    return