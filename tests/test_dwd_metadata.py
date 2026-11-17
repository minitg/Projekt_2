import tempfile
import unittest
from pathlib import Path

import pandas as pd

from audit_dwd_metadata import utc_metadata_rows


class DwdMetadataTests(unittest.TestCase):
    def test_conflicting_time_zone_or_missing_period_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.txt"
            header = "Stations_ID;Von_Datum;Bis_Datum;Stationsname;Parameter;Datenquelle (Strukturversion=SV);Zusatz-Info\n"
            row = "4104;20150101;20241231;Regensburg;R1;SYNOP;in UTC\n"
            path.write_text(header + row, encoding="latin1")
            result = utc_metadata_rows(path, ["R1"], pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31 23:00"))
            self.assertEqual(len(result), 1)
            path.write_text(header + row.replace("in UTC", "in MEZ"), encoding="latin1")
            with self.assertRaisesRegex(ValueError, "non-UTC"):
                utc_metadata_rows(path, ["R1"], pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31"))
            path.write_text(header + row.replace("20241231", "20231231"), encoding="latin1")
            with self.assertRaisesRegex(ValueError, "uninterrupted UTC coverage"):
                utc_metadata_rows(path, ["R1"], pd.Timestamp("2015-01-01"), pd.Timestamp("2024-12-31"))
