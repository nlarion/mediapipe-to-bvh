import sys
import os

def parse_bvh_hierarchy(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    joints = {}
    stack = []
    current_joint = None
    
    print(f"--- Inspecting {file_path} ---")
    
    for line in lines:
        line = line.strip()
        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split()[1]
            parent = stack[-1] if stack else None
            joints[name] = {'parent': parent, 'offset': None, 'channels': []}
            current_joint = name
            stack.append(name)
            print(f"Joint: {name}, Parent: {parent}")
            
        elif line.startswith("OFFSET"):
            parts = line.split()
            offset = [float(parts[1]), float(parts[2]), float(parts[3])]
            if current_joint:
                joints[current_joint]['offset'] = offset
                print(f"  Offset: {offset}")
                
        elif line.startswith("End Site"):
            stack.append(None)
            
        elif line.startswith("}"):
            stack.pop()
            
        elif line.startswith("MOTION"):
            break

    return joints

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_bvh_hierarchy(sys.argv[1])
    else:
        print("Usage: python check_skeleton.py <bvh_file>")
