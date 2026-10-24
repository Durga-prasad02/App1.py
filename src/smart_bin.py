"""
Smart-Bin Assistant Logic & Domain Knowledge Engine
Defines waste classes, recyclable categories, bin routing, servo flap actuation angles,
environmental carbon/energy offset metrics, and disposal guidelines.

Author: AGGIBADALA VENKATA DURGA NAGA PRASAD
Guide: Ms. G. Hema Sri
"""

from typing import Dict, Any, List

# Standard 6-Class Municipal Waste Taxonomy (aligned with TrashNet / Kaggle Waste Classification)
CLASSES: List[str] = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Organic / Food Waste"
]

# Recyclability and Bin Routing Specifications
CLASS_METADATA: Dict[str, Dict[str, Any]] = {
    "Cardboard": {
        "recyclable": True,
        "category": "Dry Recyclable (Cellulose)",
        "bin_name": "Blue Bin (Paper & Cardboard)",
        "bin_color": "#2563EB",  # Royal Blue
        "bin_icon": "📦",
        "servo_angle": 30,  # Degrees rotation for servo flap
        "co2_saved_kg": 0.45,  # per kg recycled
        "energy_saved_kwh": 1.10,
        "water_saved_liters": 7.5,
        "disposal_tip": "Flatten boxes to save bin volume. Remove plastic tape and ensure cardboard is dry and free of greasy food residue.",
        "contamination_risk": "High if stained with food grease or oil (e.g., pizza boxes belong in Organic/Compost).",
        "description": "Corrugated cartons, packaging boxes, cereal boxes, and cardboard mailing envelopes."
    },
    "Glass": {
        "recyclable": True,
        "category": "Dry Recyclable (Inorganic Glass)",
        "bin_name": "Yellow Bin (Glass & Ceramics)",
        "bin_color": "#EAB308",  # Amber / Yellow
        "bin_icon": "🍾",
        "servo_angle": 60,
        "co2_saved_kg": 0.32,
        "energy_saved_kwh": 0.85,
        "water_saved_liters": 2.0,
        "disposal_tip": "Rinse empty bottles and jars. Remove metal lids and plastic corks. Discard broken glass safely wrapped to avoid injuring waste workers.",
        "contamination_risk": "Low chemical contamination, but high hazard risk for shards.",
        "description": "Clear and colored beverage bottles, food jars, and cosmetic glassware."
    },
    "Metal": {
        "recyclable": True,
        "category": "Dry Recyclable (Ferrous & Non-Ferrous)",
        "bin_name": "Gray Bin (Metal & Cans)",
        "bin_color": "#6B7280",  # Slate Gray
        "bin_icon": "🥫",
        "servo_angle": 90,
        "co2_saved_kg": 1.85,
        "energy_saved_kwh": 4.50,
        "water_saved_liters": 15.0,
        "disposal_tip": "Rinse empty soda/soup cans. Crush aluminum cans if possible to conserve volume. Aerosol cans must be fully emptied.",
        "contamination_risk": "Low. Metal is infinitely recyclable with minimal degradation.",
        "description": "Aluminum soda cans, tin food cans, metal bottle caps, and clean aluminum foil."
    },
    "Paper": {
        "recyclable": True,
        "category": "Dry Recyclable (Paper Fiber)",
        "bin_name": "Blue Bin (Paper & Cardboard)",
        "bin_color": "#2563EB",
        "bin_icon": "📄",
        "servo_angle": 120,
        "co2_saved_kg": 0.35,
        "energy_saved_kwh": 0.90,
        "water_saved_liters": 10.0,
        "disposal_tip": "Keep dry and unsoiled. Shred confidential documents. Do not include laminated or waxed receipt paper.",
        "contamination_risk": "Wet or food-stained paper cannot be processed in pulping mills.",
        "description": "Office paper, newspapers, magazines, notebook pages, and paper flyers."
    },
    "Plastic": {
        "recyclable": True,
        "category": "Dry Recyclable (Polymers - PET, HDPE, PP)",
        "bin_name": "Orange Bin (Plastics & Polymers)",
        "bin_color": "#F97316",  # Vibrant Orange
        "bin_icon": "🧴",
        "servo_angle": 150,
        "co2_saved_kg": 1.20,
        "energy_saved_kwh": 3.20,
        "water_saved_liters": 5.0,
        "disposal_tip": "Empty and lightly rinse beverage bottles. Re-attach screw caps. Check resin codes (PET #1 and HDPE #2 are most widely accepted).",
        "contamination_risk": "Residue degrades melt quality; separate flexible multi-layer pouches if specified by municipality.",
        "description": "PET water bottles, milk jugs, shampoo flasks, detergent containers, and rigid plastic packaging."
    },
    "Organic / Food Waste": {
        "recyclable": False,  # Not mechanically recyclable, but compostable/biodegradable
        "category": "Wet Compostable (Biodegradable Biomass)",
        "bin_name": "Green Bin (Organic & Compost)",
        "bin_color": "#16A34A",  # Forest Green
        "bin_icon": "🍏",
        "servo_angle": 180,
        "co2_saved_kg": 0.60,  # Prevents anaerobic methane release in landfills
        "energy_saved_kwh": 0.40,
        "water_saved_liters": 0.0,
        "disposal_tip": "Deposit directly or use certified compostable/biodegradable bags. Suitable for aerobic composting and biogas generation.",
        "contamination_risk": "Remove all plastic stickers from fruit peels, twist-ties, and rubber bands.",
        "description": "Fruit peels, vegetable trimmings, leftover food, tea bags, coffee grounds, and garden clippings."
    }
}


def get_class_info(class_name: str) -> Dict[str, Any]:
    """Retrieve full routing metadata and environmental impact for a given class."""
    return CLASS_METADATA.get(class_name, {
        "recyclable": False,
        "category": "Residual / Landfill",
        "bin_name": "Black Bin (General Waste)",
        "bin_color": "#1F2937",
        "bin_icon": "🗑️",
        "servo_angle": 0,
        "co2_saved_kg": 0.0,
        "energy_saved_kwh": 0.0,
        "water_saved_liters": 0.0,
        "disposal_tip": "Non-recyclable general trash. Route to municipal thermal waste-to-energy or engineered landfill.",
        "contamination_risk": "N/A",
        "description": "Mixed household refuse."
    })


def calculate_batch_impact(classification_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate cumulative environmental metrics from a batch of segregated items."""
    total_co2 = 0.0
    total_energy = 0.0
    total_water = 0.0
    recyclable_count = 0
    organic_count = 0
    total_count = len(classification_history)

    for item in classification_history:
        cls_name = item.get("class_name", "")
        info = get_class_info(cls_name)
        total_co2 += info.get("co2_saved_kg", 0.0)
        total_energy += info.get("energy_saved_kwh", 0.0)
        total_water += info.get("water_saved_liters", 0.0)

        if info.get("recyclable", False):
            recyclable_count += 1
        elif "Organic" in cls_name:
            organic_count += 1

    diversion_rate = ((recyclable_count + organic_count) / max(total_count, 1)) * 100.0

    return {
        "total_items": total_count,
        "recyclable_items": recyclable_count,
        "organic_items": organic_count,
        "co2_saved_kg": round(total_co2, 2),
        "energy_saved_kwh": round(total_energy, 2),
        "water_saved_liters": round(total_water, 2),
        "landfill_diversion_rate_pct": round(diversion_rate, 1)
    }
