#!/usr/bin/env python3
"""BIS Domain 54 Medical Equipment Compliance & Gap Analysis Tool.

Audits medical devices, hospital equipment, and disposable surgical products
against BIS Indian Standards, MHD draft committees, and national certification schemes.
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

from rag_engine import BISRAGEngine
from bis_knowledge import SCHEMES, REVIEWED


COMMITTEES_MAP = {
    "MHD 1": {
        "title": "Surgical Instruments",
        "scope": "Manual & powered surgical instruments, scalpels, dissecting scissors, forceps, needle holders, retractors.",
        "key_standards": ["IS 3642", "MHD 1 (34505) WC", "MHD 1 (34489) WC"]
    },
    "MHD 4": {
        "title": "Orthopaedic Instruments and Implants",
        "scope": "Internal fixation devices, bone screws, plates, joint prostheses, biocompatible titanium and cobalt alloys.",
        "key_standards": ["ISO 5832", "IS 10250", "IS 13422"]
    },
    "MHD 8": {
        "title": "Dentistry and Dental Materials",
        "scope": "Dental restorative polymers, acrylic resins, impression materials, dental cements, rotary instruments.",
        "key_standards": ["MHD 8 (34260) WC", "ISO 4049", "ISO 1567"]
    },
    "MHD 10": {
        "title": "Medical Laboratory Equipment",
        "scope": "In vitro diagnostic instruments, vortex mixers, laboratory centrifuges, incubators, blood mixers.",
        "key_standards": ["MHD 10 (28479) WC", "IEC 61010-2-010", "IS 16046"]
    },
    "MHD 12": {
        "title": "Hospital Equipment and Surgical Disposable Products",
        "scope": "Hypodermic glass & plastic syringes, volumetric infusion sets, needle-free injectors, sterile vials, catheters.",
        "key_standards": ["MHD 12 (34618) WC", "MHD 12 (34619) WC", "MHD 12 (34620) WC", "MHD 12 (34622) WC", "MHD 12 (34623) WC"]
    },
    "MHD 13": {
        "title": "Veterinary Surgical Instruments",
        "scope": "Veterinary instruments, pyro-puncture with copper needle, animal castration and obstetrical forceps.",
        "key_standards": ["MHD 13 (34744) WC", "IS 7332"]
    },
    "MHD 15": {
        "title": "Anaesthetic and Resuscitation Equipment",
        "scope": "Breathing systems, oxygen delivery systems, ventilators, medical suction equipment, airway humidifiers.",
        "key_standards": ["ISO 80601-2-12", "IS 13107", "IS 14371"]
    },
    "MHD 17": {
        "title": "Health Informatics & Medical Software",
        "scope": "DICOM workflow, PACS image management, telehealth data standards, medical device cybersecurity and interoperability.",
        "key_standards": ["MHD 17 (34655) WC", "ISO 12052", "ISO 62304"]
    },
    "MHD 20": {
        "title": "Medical Biotechnology and Additive Manufacturing",
        "scope": "3D printed patient-specific implants, biomaterials, scaffolds for in vivo tissue regeneration, additive manufacturing.",
        "key_standards": ["MHD 20 (32899) WC", "ISO/ASTM 52900", "ISO 10993"]
    }
}


class MedicalComplianceAuditor:
    """Performs structured regulatory and gap analysis for BIS Domain 54 medical equipment."""

    def __init__(self, index_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent
        target_index = index_path or (base_dir / "corpus_index.pkl")
        if not target_index.exists():
            raise FileNotFoundError(f"Index file not found: {target_index}. Run index_model.py first.")
        self.engine = BISRAGEngine(target_index)

    def identify_committee(self, query: str, retrieved_passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identifies the primary Sectional Committee under Domain 54."""
        query_upper = query.upper()
        # Direct query matching using word boundaries
        for code, info in COMMITTEES_MAP.items():
            pattern = rf"\b{re.escape(code)}\b"
            if re.search(pattern, query_upper) or info["title"].upper() in query_upper:
                return {"code": code, **info}

        # Check top retrieved passage
        if retrieved_passages:
            raw_comm = retrieved_passages[0].get("committee", "").strip().upper()
            # Normalize e.g. "MHD 01" to "MHD 1"
            norm_comm = re.sub(r"MHD\s*0?(\d+)", r"MHD \1", raw_comm)
            if norm_comm in COMMITTEES_MAP:
                return {"code": norm_comm, **COMMITTEES_MAP[norm_comm]}
            for code, info in COMMITTEES_MAP.items():
                if code == norm_comm:
                    return {"code": code, **info}

        # Fallback default
        return {"code": "MHD 12", **COMMITTEES_MAP["MHD 12"]}

    def determine_certification_route(self, query: str, committee_code: str, origin: str = "domestic") -> Dict[str, Any]:
        """Maps equipment to applicable BIS certification scheme."""
        q = query.lower()
        if origin.lower() == "foreign":
            return {
                "scheme_id": "fmcs",
                "name": SCHEMES["fmcs"]["title"],
                "portal": "Manakonline FMCS Portal",
                "air_required": True,
                "notes": "Foreign manufacturers must nominate an Authorized Indian Representative (AIR) residing in India."
            }

        # Health informatics / Active electronic devices with wireless/IT
        if committee_code == "MHD 17" or any(kw in q for kw in ["software", "dicom", "telemetry", "wireless", "pacs"]):
            return {
                "scheme_id": "crs",
                "name": SCHEMES["crs"]["title"],
                "portal": "CRS Portal (crsbis.in)",
                "air_required": False,
                "notes": "Testing at BIS-recognized CRS lab followed by online registration without mandatory prior factory inspection."
            }

        # Standard physical medical equipment and disposables (Default: Scheme I ISI Mark)
        return {
            "scheme_id": "isi",
            "name": SCHEMES["isi"]["title"],
            "portal": "e-BIS / Manakonline",
            "air_required": False,
            "notes": "Requires factory quality assessment, in-house testing facility audit, independent lab sample testing, and surveillance."
        }

    def generate_checklist(self, query: str, committee_code: str) -> List[Dict[str, str]]:
        """Generates standard mandatory technical clauses to audit."""
        checklist = [
            {
                "category": "Material & Chemical Purity",
                "requirement": "Conformity to medical-grade metallurgy/polymers, heavy metals limits, extractables & leachables testing.",
                "verification_method": "Spectrometry, ICP-MS (extractable tungsten/metals), FTIR polymer fingerprinting."
            },
            {
                "category": "Biocompatibility & Safety",
                "requirement": "Biological evaluation under ISO 10993 / in-vivo criteria (cytotoxicity, sensitization, irritation, systemic toxicity).",
                "verification_method": "Certified GLP laboratory biological evaluation report."
            },
            {
                "category": "Sterility & Packaging Integrity",
                "requirement": "Validation of sterilization cycle (EO, gamma radiation, steam) and sterile barrier shelf-life (ISO 11607).",
                "verification_method": "Bioburden assay, sterility challenge, dye penetration, accelerated aging test."
            },
            {
                "category": "Mechanical & Functional Performance",
                "requirement": "Dimensional tolerances, leak resistance, flow rate accuracy, corrosion resistance, tensile strength.",
                "verification_method": "Calibrated pressure test, boiling water corrosion test, automated tensile testing."
            },
            {
                "category": "Marking, Labelling & UDI",
                "requirement": "Unique Device Identifier (UDI), batch number, manufacturing & expiry dates, sterile symbol, standard reference.",
                "verification_method": "Visual inspection, solvent-rub rub resistance test, barcode/QR readability."
            }
        ]

        if committee_code in ["MHD 10", "MHD 17"]:
            checklist.append({
                "category": "Electrical Safety & EMC",
                "requirement": "Compliance with IEC 61010-1 / IEC 60601-1 electrical safety and electromagnetic compatibility.",
                "verification_method": "Dielectric withstand, earth leakage current, radiated/conducted emissions testing."
            })

        return checklist

    def audit(self, query: str, origin: str = "domestic", top_k: int = 4) -> Dict[str, Any]:
        """Runs end-to-end regulatory compliance audit and gap analysis."""
        qa_result = self.engine.answer_question(query, top_k=top_k)
        retrieved = qa_result.get("retrieved_passages", [])
        committee = self.identify_committee(query, retrieved)
        cert_route = self.determine_certification_route(query, committee["code"], origin=origin)
        checklist = self.generate_checklist(query, committee["code"])

        # Compile referenced standards
        standards_identified = []
        for p in retrieved:
            standards_identified.append({
                "document_no": p.get("document_no"),
                "title": p.get("title"),
                "committee": p.get("committee"),
                "page": p.get("page_number"),
                "relevance_score": p.get("score")
            })

        gap_recommendations = [
            f"1. Retrieve the official BIS draft document for {committee['code']} to verify exact dimensional tables and tolerances.",
            f"2. Confirm manufacturing testing capabilities for {checklist[0]['category'].lower()} prior to BIS application.",
            f"3. Submit pre-compliance samples to a BIS-recognized or NABL-accredited medical testing laboratory.",
            f"4. Prepare documentation under {cert_route['name']} via {cert_route['portal']}.",
            "5. Implement complete batch traceability including Device Master Record (DMR) and Device History Record (DHR)."
        ]

        return {
            "query": query,
            "origin": origin,
            "reviewed_date": REVIEWED,
            "committee": committee,
            "certification_route": cert_route,
            "applicable_standards": standards_identified,
            "extracted_answer": qa_result.get("answer", ""),
            "compliance_checklist": checklist,
            "gap_action_plan": gap_recommendations
        }

    def format_markdown(self, audit_result: Dict[str, Any]) -> str:
        """Formats the audit result into a clean GitHub Flavored Markdown report."""
        res = audit_result
        lines = []
        lines.append(f"# BIS Domain 54 Medical Compliance & Audit Matrix")
        lines.append(f"**Target Equipment / Query:** `{res['query']}`  ")
        lines.append(f"**Manufacture Origin:** `{res['origin'].capitalize()}` | **Audit Base:** BIS Medical Domain 54  \n")

        lines.append("## 1. Sectional Committee & Scope")
        comm = res["committee"]
        lines.append(f"- **Committee:** **{comm['code']} — {comm['title']}**")
        lines.append(f"- **Scope:** {comm['scope']}")
        lines.append(f"- **Key Standards in Committee:** {', '.join(comm['key_standards'])}\n")

        lines.append("## 2. Applicable Certification Route")
        route = res["certification_route"]
        lines.append(f"- **Certification Scheme:** **{route['name']}**")
        lines.append(f"- **Application Portal:** {route['portal']}")
        lines.append(f"- **Foreign AIR Mandatory:** {'Yes' if route['air_required'] else 'No'}")
        lines.append(f"- **Requirements:** {route['notes']}\n")

        lines.append("## 3. Top Matched BIS Standards & Drafts")
        lines.append("| Standard No. | Title | Committee | Page | Match Score |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for std in res["applicable_standards"]:
            lines.append(f"| **{std['document_no']}** | {std['title']} | {std['committee']} | Pg {std['page']} | `{std['relevance_score']}` |")
        lines.append("")

        lines.append("## 4. Technical Compliance Checklist")
        lines.append("| Audit Category | Regulatory Requirement | Verification Method |")
        lines.append("| :--- | :--- | :--- |")
        for item in res["compliance_checklist"]:
            lines.append(f"| **{item['category']}** | {item['requirement']} | `{item['verification_method']}` |")
        lines.append("")

        lines.append("## 5. Pre-Audit Action Plan & Gap Recommendations")
        for rec in res["gap_action_plan"]:
            lines.append(f"- {rec}")
        lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BIS Domain 54 Medical Equipment Compliance Auditor")
    parser.add_argument("query", nargs="?", help="Product or standard query (e.g. 'disposable infusion sets')")
    parser.add_argument("--query", "-q", dest="query_opt", help="Alternative query flag")
    parser.add_argument("--origin", choices=["domestic", "foreign"], default="domestic", help="Manufacturing origin")
    parser.add_argument("--format", choices=["json", "md", "text"], default="text", help="Output format")
    parser.add_argument("--output", "-o", help="File to write output to")
    parser.add_argument("--list-committees", action="store_true", help="List all BIS Domain 54 Sectional Committees")
    args = parser.parse_args()

    if args.list_committees:
        print("\n" + "=" * 70)
        print("  BIS DOMAIN 54 — MEDICAL EQUIPMENT & HOSPITAL PLANNING COMMITTEES")
        print("=" * 70)
        for code, info in sorted(COMMITTEES_MAP.items()):
            print(f"\n[{code}] {info['title']}")
            print(f"  Scope: {info['scope']}")
            print(f"  Key Standards: {', '.join(info['key_standards'])}")
        print("\n" + "=" * 70)
        return

    query = args.query or args.query_opt
    if not query:
        parser.print_help()
        sys.exit(1)

    auditor = MedicalComplianceAuditor()
    audit_data = auditor.audit(query, origin=args.origin)

    if args.format == "json":
        output_str = json.dumps(audit_data, indent=2, ensure_ascii=False)
    elif args.format == "md":
        output_str = auditor.format_markdown(audit_data)
    else:
        # User friendly text output
        md_text = auditor.format_markdown(audit_data)
        output_str = md_text

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        print(f"Audit report saved to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
