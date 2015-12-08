'''
Created on Aug 5, 2015

@author: Yingxiong
'''

directory = 'D:\data\\simdb\\exdata\\tensile_tests\\buttstrap_clamping\\2015-08-03_TTb-2C-14mm-0-3300SBR_cyc-Aramis2d\\'
file = 'TTb-2C-14mm-0-3300SBR-V1_cyc-Aramis2d.csv'
file_path = directory + file

with open(file_path, 'r') as fin:
    with open(file_path.replace('.csv', '-force.csv'), 'w') as force_out, open(file_path.replace('.csv', '-disp.csv'), 'w') as disp_out:
        next(fin)
        next(fin)
        for line in fin:
            line_dot = line.replace(',', '.')
#             raw = line_dot.split(';')
#             disp = -0.5 * (float(raw[5]) + float(raw[6]))
            try:
                raw = line_dot.split(';')
                # avg_disp = -(vor+(re+li)/2)/2
#                 disp = -0.5 * \
#                     (float(raw[4]) + 0.5 * (float(raw[5]) + float(raw[6])))

                disp = -0.5 * (float(raw[5]) + float(raw[6]))
                disp_out.write("%.3f" % disp + '\n')
                force_out.write("%.3f" % float(raw[1]) + '\n')
            except:
                print 'cannot recognize:', line

fin.close()
force_out.close()
disp_out.close()
