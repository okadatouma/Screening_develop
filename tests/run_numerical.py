"""数値テストだけを実行し、依存不足でも各メソッドのSKIPを記録する。"""
import argparse
import ast
import json
import sys
from pathlib import Path
import unittest

from tests.baseline import metadata
from tests.support import ROOT, require_numerics


def numerical_tests():
    for path in sorted(ROOT.joinpath('tests').glob('test_*.py')):
        if path.stem in ('test_legacy', 'test_reproducibility'):
            continue
        for cls in ast.parse(path.read_text()).body:
            if isinstance(cls, ast.ClassDef):
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef) and method.name.startswith('test_'):
                        yield f'tests.{path.stem}.{cls.name}.{method.name}'


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def record(self, test, status, detail=''):
        self.records.append(dict(test=test.id(), status=status, detail=detail, environment='execution'))

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, 'PASS')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, 'FAIL', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, 'ERROR', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, 'SKIP', reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.record(test, 'XFAIL', self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.record(test, 'XPASS', '既知問題の期待失敗が成立しないため要確認')

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.record(subtest, 'FAIL' if issubclass(err[0], test.failureException) else 'ERROR',
                        self._exc_info_to_string(err, test))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='新規JSONファイル（既存ファイルを上書きしない）')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('既存ファイルは上書きしません')
    names = list(numerical_tests())
    try:
        require_numerics()
    except unittest.SkipTest as error:
        records = [dict(test=name, status='SKIP', detail=str(error), environment='execution') for name in names]
        success = False
    else:
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
        records, success = result.records, result.wasSuccessful()
    report = dict(environment=metadata(), requested_tests=names, results=records,
                  numerical_execution_complete=success and not any(r['status'] == 'SKIP' for r in records))
    if 'jax' in sys.modules:
        import jax
        report['environment']['runtime'] = dict(backend=jax.default_backend(),
            devices=[str(d) for d in jax.devices()], x64=bool(jax.config.x64_enabled),
            matmul_precision=str(jax.config.jax_default_matmul_precision))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    for record in records:
        print(record['status'], record['test'])
    # 全件SKIPでも正常終了扱いにはしない。
    raise SystemExit(0 if report['numerical_execution_complete'] else 2)


if __name__ == '__main__':
    main()
