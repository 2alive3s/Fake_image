import random

mission1_result = open('/usr/junny/fake_photo/team_haechi/result/mission2_result.txt', 'r')
mission1_testlabel = open('/usr/junny/fake_photo/AUROC/mission2_preround.txt', 'w')
for row in mission1_result :
    id = row.split(',')[0]
    mission1_testlabel.write(id+','+str(random.randint(0,1))+'\n')

