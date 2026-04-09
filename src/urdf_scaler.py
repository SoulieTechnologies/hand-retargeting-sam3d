import xml.etree.ElementTree as ET

def scale_urdf_z_only(urdf_path, output_path, scale_factors):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    is_global = "GLOBAL" in scale_factors
    global_factor = scale_factors.get("GLOBAL", 1.0)

    def is_base(name):
        return any(x in name for x in ['Carpals', 'Tower', 'ForeArm'])

    # --- STAGE 2: Scale the Joint Origins (Z-Axis Length Only) ---
    for joint in root.findall('joint'):
        child_tag = joint.find('child')
        parent_tag = joint.find('parent') 
        
        if child_tag is not None and parent_tag is not None:
            child_link_name = child_tag.attrib['link']
            parent_link_name = parent_tag.attrib['link']
            
            origin = joint.find('origin')
            if origin is not None and 'xyz' in origin.attrib:
                x, y, z = [float(v) for v in origin.attrib['xyz'].split()]
                
                if is_global:
                    if not is_base(parent_link_name) and not is_base(child_link_name):
                        new_z = z * global_factor
                        origin.attrib['xyz'] = f"{x} {y} {new_z}"
                else:
                    for prefix, factor in scale_factors.items():
                        if child_link_name.startswith(prefix) and parent_link_name.startswith(prefix):
                            new_z = z * factor
                            origin.attrib['xyz'] = f"{x} {y} {new_z}"

    # --- STAGE 3: Stretch the 3D Meshes (Z-Axis Length Only) ---
    for link in root.findall('link'):
        link_name = link.attrib['name']
        
        for mesh in link.findall('.//geometry/mesh'):
            if 'scale' in mesh.attrib:
                sx, sy, sz = [float(v) for v in mesh.attrib['scale'].split()]
                
                if not is_base(link_name):
                    if is_global:
                        mesh.attrib['scale'] = f"{sx} {sy} {sz * global_factor}"
                    else:
                        for prefix, factor in scale_factors.items():
                            if link_name.startswith(prefix):
                                mesh.attrib['scale'] = f"{sx} {sy} {sz * factor}"

    tree.write(output_path)
    print(f"\nSaved perfectly anchored URDF to: {output_path}")

if __name__ == "__main__":
    base_urdf = "../assets/orcahand/orcahand_right.urdf"
    scaled_urdf = "../assets/orcahand/orcahand_scaled.urdf"
    
    # Mode A: Stretch ALL finger lengths globally (Anchored safely to the palm & arm)
    my_factors = {"GLOBAL": 1.5} 
    
    scale_urdf_z_only(base_urdf, scaled_urdf, my_factors)