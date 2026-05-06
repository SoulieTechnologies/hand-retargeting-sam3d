import xml.etree.ElementTree as ET

def scale_and_shift_urdf(urdf_path, output_path, global_factor, z_shift):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    root_link_name = "right_hand_C_MC"
    
    # Dans ce script, on gère les dictionnaires (ex: {"GLOBAL": 0.9}) 
    # ou les simples floats (ex: 0.9)
    if isinstance(global_factor, dict):
        scale_factors = global_factor
        global_f = scale_factors.get("GLOBAL", 1.0)
    else:
        scale_factors = {"GLOBAL": global_factor}
        global_f = global_factor

    # --- ETAPE 1 : Décaler et Mettre à l'échelle TOUTES les origines de Links (Visuel/Collision) ---
    for link in root.findall('link'):
        link_name = link.attrib.get('name')
        
        # 1. Corriger les translations (origin) pour les visuels et collisions
        for origin in link.findall('.//origin'):
            if 'xyz' in origin.attrib:
                x, y, z = [float(v) for v in origin.attrib['xyz'].split()]
                
                # Si c'est la paume, on applique le décalage (shift) EN PREMIER
                if link_name == root_link_name:
                    z += z_shift
                
                # ENSUITE, on applique l'échelle globale pour que l'espacement reste proportionnel
                origin.attrib['xyz'] = f"{x * global_f} {y * global_f} {z * global_f}"
        
        # 2. Mettre à l'échelle le volume des Mesh 3D
        for mesh in link.findall('.//geometry/mesh'):
            sx, sy, sz = [float(v) for v in mesh.attrib.get('scale', '1 1 1').split()]
            current_sz = sz * global_f
            
            # (Optionnel) Si on a des échelles par doigt
            for prefix, f in scale_factors.items():
                if prefix != "GLOBAL" and link_name.startswith(prefix):
                    current_sz *= f
            
            mesh.attrib['scale'] = f"{sx * global_f} {sy * global_f} {current_sz}"

    # --- ETAPE 2 : Décaler et Mettre à l'échelle TOUS les Joints (Le Squelette) ---
    for joint in root.findall('joint'):
        parent_link = joint.find('parent').attrib.get('link')
        child_link = joint.find('child').attrib.get('link')
        origin = joint.find('origin')
        
        if origin is not None and 'xyz' in origin.attrib:
            x, y, z = [float(v) for v in origin.attrib['xyz'].split()]
            
            # Si le doigt est attaché à la paume, on applique le même décalage
            if parent_link == root_link_name:
                z += z_shift
            
            # ENSUITE on applique l'échelle globale (Le squelette et le visuel restent ainsi soudés)
            nx, ny, nz = x * global_f, y * global_f, z * global_f
            
            # (Optionnel) Si on a des échelles par doigt
            for prefix, f in scale_factors.items():
                if prefix != "GLOBAL" and child_link is not None and child_link.startswith(prefix) and parent_link.startswith(prefix):
                    nz *= f
            
            origin.attrib['xyz'] = f"{nx} {ny} {nz}"

    tree.write(output_path)
    print(f"URDF Réparé et Sauvegardé : {output_path}")

# Si tu lances le script tout seul pour tester
if __name__ == "__main__":
    base = "../assets/sharpa_hand/wave_01/right_sharpa_wave/right_sharpa_wave.urdf"
    scaled = "../assets/sharpa_hand/wave_01/right_sharpa_wave/right_sharpa_wave_scaled.urdf"
    scale_and_shift_urdf(base, scaled, 0.9, -0.15)