import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC=importlib.util.spec_from_file_location('preserve',Path(__file__).with_name('preserve.py'))
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.results=self.root/'original';self.here=self.root/'archive'
        self.results.mkdir();self.here.mkdir()
        (self.results/'trial').mkdir();(self.results/'trial/raw.csv').write_text('original raw\n')
        row=dict(path='trial/raw.csv',bytes=(self.results/'trial/raw.csv').stat().st_size,
                 sha256=p.sha(self.results/'trial/raw.csv'))
        p.publish(self.here/'raw_inventory.jsonl',(json.dumps(row)+'\n').encode())
        p.save(self.here/'manifest.json',dict(raw_inventory_sha256=p.sha(self.here/'raw_inventory.jsonl'),
            raw_file_count=1,raw_roots=['trial'],raw_logical_bytes=row['bytes']))

    def test_original_verification_is_readonly(self):
        before=(self.results/'trial/raw.csv').read_bytes()
        with mock.patch.object(p,'HERE',self.here):p.verify(self.results)
        self.assertEqual((self.results/'trial/raw.csv').read_bytes(),before)

    def test_corruption_is_rejected(self):
        (self.results/'trial/raw.csv').write_text('wrong raw data')
        with mock.patch.object(p,'HERE',self.here),self.assertRaises(AssertionError):p.verify(self.results)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):p.publish(self.results/'trial/raw.csv',b'replace')

    def test_fresh_backup_verified(self):
        with mock.patch.object(p,'HERE',self.here),mock.patch.object(p,'RESULTS',self.results):
            p.backup(self.root/'backup')
        self.assertTrue((self.root/'backup/BACKUP_VERIFIED.json').exists())
        self.assertEqual(p.sha(self.root/'backup/trial/raw.csv'),p.sha(self.results/'trial/raw.csv'))

    def test_existing_destination_refused(self):
        dest=self.root/'backup';dest.mkdir()
        with mock.patch.object(p,'HERE',self.here),self.assertRaises(AssertionError):p.backup(dest)

    def test_insufficient_space_refused_before_copy(self):
        with mock.patch.object(p,'HERE',self.here),mock.patch.object(p.shutil,'disk_usage',return_value=mock.Mock(free=0)),self.assertRaises(AssertionError):
            p.backup(self.root/'backup')
        self.assertFalse((self.root/'backup').exists())


if __name__=='__main__':unittest.main()
