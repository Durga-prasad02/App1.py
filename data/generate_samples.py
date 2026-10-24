"""
Generate Sample Images for Waste Segregation Prototype
Creates categorized test samples for Cardboard, Glass, Metal, Paper, Plastic, and Organic Waste.

Author: AGGIBADALA VENKATA DURGA NAGA PRASAD
Guide: Ms. G. Hema Sri
"""

import os
from PIL import Image, ImageDraw

CATEGORIES = [
    ("Cardboard", "cardboard_box.png", "#C19A6B"),
    ("Glass", "glass_bottle.png", "#A7D7C5"),
    ("Metal", "soda_can.png", "#9CA3AF"),
    ("Paper", "office_paper.png", "#FFFFFF"),
    ("Plastic", "water_bottle.png", "#60A5FA"),
    ("Organic", "fruit_peels.png", "#84CC16")
]


def generate_samples(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    
    for category, filename, primary_color in CATEGORIES:
        cat_dir = os.path.join(base_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        img_path = os.path.join(cat_dir, filename)
        
        img = Image.new("RGB", (320, 320), color="#F8FAFC")
        draw = ImageDraw.Draw(img)
        
        if category == "Cardboard":
            # Cardboard carton
            draw.rectangle([50, 70, 270, 250], fill="#C19A6B", outline="#8B5A2B", width=4)
            draw.line([50, 160, 270, 160], fill="#8B5A2B", width=3)  # Tape
            draw.rectangle([100, 180, 220, 220], fill="#E5D3B3")
            draw.text((120, 195), "CARDBOARD", fill="#5C4033")
            
        elif category == "Glass":
            # Translucent glass bottle
            draw.rectangle([140, 40, 180, 80], fill="#99D5C9", outline="#4D8B7E", width=3)
            draw.polygon([(140, 80), (180, 80), (230, 130), (90, 130)], fill="#99D5C9", outline="#4D8B7E")
            draw.rectangle([90, 130, 230, 280], fill="#B2E2D8", outline="#4D8B7E", width=4)
            draw.line([115, 150, 115, 260], fill="#FFFFFF", width=4)  # Specular reflection
            draw.text((130, 190), "GLASS BOTTLE", fill="#1F4E44")
            
        elif category == "Metal":
            # Aluminum can
            draw.ellipse([100, 50, 220, 90], fill="#D1D5DB", outline="#4B5563", width=3)
            draw.rectangle([100, 70, 220, 250], fill="#9CA3AF", outline="#4B5563", width=3)
            draw.ellipse([100, 230, 220, 270], fill="#6B7280", outline="#4B5563", width=3)
            draw.rectangle([100, 120, 220, 190], fill="#EF4444")
            draw.text((125, 145), "ALUMINUM CAN", fill="#FFFFFF")
            
        elif category == "Paper":
            # Document / printed paper
            draw.polygon([(70, 40), (230, 40), (260, 70), (260, 270), (70, 270)], fill="#FFFFFF", outline="#94A3B8", width=3)
            draw.polygon([(230, 40), (260, 70), (230, 70)], fill="#E2E8F0", outline="#94A3B8")
            for y in range(90, 250, 22):
                draw.line([90, y, 240, y], fill="#CBD5E1", width=3)
            draw.text((90, 55), "DOCUMENT", fill="#0F172A")
            
        elif category == "Plastic":
            # Water bottle
            draw.rectangle([140, 35, 180, 60], fill="#2563EB")  # Cap
            draw.polygon([(140, 60), (180, 60), (225, 105), (95, 105)], fill="#93C5FD", outline="#2563EB", width=3)
            draw.rectangle([95, 105, 225, 270], fill="#BFDBFE", outline="#2563EB", width=3)
            draw.rectangle([95, 150, 225, 205], fill="#3B82F6")
            draw.text((120, 170), "PLASTIC BOTTLE", fill="#FFFFFF")
            draw.line([115, 120, 115, 250], fill="#FFFFFF", width=3)
            
        else:
            # Organic / food peels
            draw.ellipse([90, 85, 230, 235], fill="#84CC16", outline="#3F6212", width=4)
            draw.rectangle([155, 45, 165, 90], fill="#78350F")  # Stem
            draw.polygon([(165, 60), (205, 45), (185, 75)], fill="#15803D")  # Leaf
            draw.ellipse([120, 120, 150, 150], fill="#65A30D")
            draw.text((120, 185), "ORGANIC WASTE", fill="#14532D")
            
        img.save(img_path)
        print(f"Generated sample: {img_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(current_dir, "sample_images")
    generate_samples(sample_dir)
