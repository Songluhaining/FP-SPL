"""真实Java插桩回归：构造器、工厂返回、字段读取及两次执行的共享语句。"""
import os
from pathlib import Path

import pytest

from cfp_spl.brfl_runtime import BrflRuntime
from cfp_spl.domain import Configuration
from cfp_spl.factor_graph import causal_statement_keys, load_factor_graph
from cfp_spl.java_tests import discover_tests


def test_constructor_return_dependencies_and_joint_observations(tmp_path):
    java_home = Path(os.environ.get('JAVA_HOME', 'E:/Program Files/Java/jdk-1.8'))
    if not (java_home / 'bin/java.exe').exists():
        pytest.skip('需要Java 8开发工具包执行真实插桩回归')
    variant = tmp_path / 'variant'
    for folder in ('src/main', 'test/main', 'coverage'):
        (variant / folder).mkdir(parents=True)
    (variant / 'src/main/Probe.java').write_text(
        'package main; public class Probe { private int value; '
        'public Probe(int n){value=n;} public int get(){return value;} '
        'public static Probe make(int n){return new Probe(n);} }', encoding='utf-8')
    (variant / 'test/main/ProbeTest.java').write_text(
        'package main; import org.junit.Test; import static org.junit.Assert.*; '
        'public class ProbeTest { '
        '@Test public void testPass(){Probe x=Probe.make(7); assertEquals(7,x.get());} '
        '@Test public void testFail(){Probe x=Probe.make(7); assertEquals(8,x.get());} }', encoding='utf-8')
    (variant / 'coverage/spectrum_passed_coverage.xml').write_text(
        '<coverage><package name="main"><file name="main.Probe" path="main/Probe.java">'
        '<line num="1" featureClass="Base.main.Probe" featureLineNum="1"/>'
        '</file></package></coverage>', encoding='utf-8')
    config = Configuration('C', (1,), ('Base',), 'passed', variant)
    runtime = BrflRuntime(tmp_path / 'runtime', java_home)
    runtime.build_brfl()
    build = runtime.compile_configuration(config, tmp_path / 'build')
    tests = {test.method_name: test for test in discover_tests(variant)}
    executions = [(config, tests[name], runtime.execute(config, tests[name], tmp_path / 'execution', build))
                  for name in ('testPass', 'testFail')]
    assert [e.outcome for _, _, e in executions] == ['passed', 'failed']
    local_graphs = [load_factor_graph(e.graph_path) for _, _, e in executions]
    for graph in local_graphs:
        paths = causal_statement_keys(graph)
        assert any('main.Probe:<init>:' in key for key in paths)
        assert any('main.Probe:get:' in key for key in paths)
    joint = load_factor_graph(runtime.infer_joint(executions, tmp_path / 'joint'))
    mapped = [s for s in joint.statements.values() if s.stable_key == 'Base.main.Probe:1']
    assert len(mapped) == 1
    assert len(joint.values) == sum(len(g.values) for g in local_graphs)
    observed = {v['observed_value'] for v in joint.values.values() if v['observed']}
    assert observed == {True, False}
    assert any(s.observed and s.correctness > .99 for s in joint.statements.values())
