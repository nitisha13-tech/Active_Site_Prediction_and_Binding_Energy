import function_H_bond_SB as my_code
import sys

H_bond_list= my_code.get_H_bond(sys.argv[1], sys.argv[2])

#print(H_bond_list)

Salt_bridge_list= my_code.salt_bridge(sys.argv[1], sys.argv[2])

#print(Salt_bridge_list)

for i in H_bond_list:
        print(i)


for i in Salt_bridge_list:
        print(i)
