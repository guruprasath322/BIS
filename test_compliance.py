import unittest
from pathlib import Path
from compliance_checker import MedicalComplianceAuditor, COMMITTEES_MAP


class MedicalComplianceAuditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = Path(__file__).resolve().parent
        cls.auditor = MedicalComplianceAuditor(base_dir / "corpus_index.pkl")

    def test_committees_map_integrity(self):
        expected_committees = {"MHD 1", "MHD 4", "MHD 8", "MHD 10", "MHD 12", "MHD 13", "MHD 15", "MHD 17", "MHD 20"}
        self.assertTrue(expected_committees.issubset(set(COMMITTEES_MAP.keys())))
        for code, info in COMMITTEES_MAP.items():
            self.assertIn("title", info)
            self.assertIn("scope", info)
            self.assertIn("key_standards", info)
            self.assertTrue(len(info["key_standards"]) > 0)

    def test_identify_committee_by_query(self):
        # Explicit committee or title query
        comm_mhd1 = self.auditor.identify_committee("MHD 1 surgical scalpel", [])
        self.assertEqual(comm_mhd1["code"], "MHD 1")

        comm_mhd17 = self.auditor.identify_committee("Health Informatics & Medical Software", [])
        self.assertEqual(comm_mhd17["code"], "MHD 17")

        # Passages-based identification
        passages = [{"committee": "MHD 10", "document_no": "MHD 10 (28479) WC"}]
        comm_from_passages = self.auditor.identify_committee("vortex mixer speed", passages)
        self.assertEqual(comm_from_passages["code"], "MHD 10")

    def test_certification_routes(self):
        # Domestic physical equipment -> Scheme I ISI
        route_isi = self.auditor.determine_certification_route("surgical scissors", "MHD 1", origin="domestic")
        self.assertEqual(route_isi["scheme_id"], "isi")
        self.assertFalse(route_isi["air_required"])

        # Foreign manufacturer -> FMCS
        route_fmcs = self.auditor.determine_certification_route("infusion sets", "MHD 12", origin="foreign")
        self.assertEqual(route_fmcs["scheme_id"], "fmcs")
        self.assertTrue(route_fmcs["air_required"])

        # Health informatics / Digital -> CRS
        route_crs = self.auditor.determine_certification_route("DICOM PACS telemetry software", "MHD 17", origin="domestic")
        self.assertEqual(route_crs["scheme_id"], "crs")
        self.assertFalse(route_crs["air_required"])

    def test_generate_checklist(self):
        checklist_mhd12 = self.auditor.generate_checklist("disposable syringe", "MHD 12")
        categories_mhd12 = [c["category"] for c in checklist_mhd12]
        self.assertIn("Material & Chemical Purity", categories_mhd12)
        self.assertIn("Biocompatibility & Safety", categories_mhd12)
        self.assertIn("Sterility & Packaging Integrity", categories_mhd12)
        self.assertNotIn("Electrical Safety & EMC", categories_mhd12)

        checklist_mhd10 = self.auditor.generate_checklist("laboratory vortex mixer", "MHD 10")
        categories_mhd10 = [c["category"] for c in checklist_mhd10]
        self.assertIn("Electrical Safety & EMC", categories_mhd10)

    def test_audit_structure(self):
        result = self.auditor.audit("infusion sets for volumetric controllers", origin="domestic")
        self.assertIn("query", result)
        self.assertIn("committee", result)
        self.assertIn("certification_route", result)
        self.assertIn("applicable_standards", result)
        self.assertIn("compliance_checklist", result)
        self.assertIn("gap_action_plan", result)

        self.assertEqual(result["committee"]["code"], "MHD 12")
        self.assertTrue(len(result["applicable_standards"]) > 0)
        self.assertTrue(any("34620" in s["document_no"] for s in result["applicable_standards"]))

    def test_format_markdown(self):
        result = self.auditor.audit("needle-free injection systems", origin="domestic")
        md = self.auditor.format_markdown(result)
        self.assertIn("# BIS Domain 54 Medical Compliance & Audit Matrix", md)
        self.assertIn("## 1. Sectional Committee & Scope", md)
        self.assertIn("## 2. Applicable Certification Route", md)
        self.assertIn("## 3. Top Matched BIS Standards & Drafts", md)
        self.assertIn("## 4. Technical Compliance Checklist", md)
        self.assertIn("## 5. Pre-Audit Action Plan & Gap Recommendations", md)


if __name__ == "__main__":
    unittest.main()
