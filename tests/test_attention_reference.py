"""Synthetic, model-free tests for an exact private H286 reference binding."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from attention_probe import reference as REF


def documents():
    # Generated here; these are not the user's private neuron identities.
    coordinates = [[i % 28, 101 + i * 7] for i in range(286)]
    model = dict(repo=REF.MODEL_REPO, revision=REF.MODEL_REVISION,
                 sha256={"synthetic-model.safetensors": "a"*64, "synthetic-tokenizer.json": "b"*64})
    provenance = dict(models={REF.MODEL_ID: model}, h_policy="htq_v2", post_hoc=True, amendment=2, mock=False)
    common = dict(h_policy="htq_v2", post_hoc=True, amendment=2, H_op_name="H_tq", provenance=provenance)
    hn = dict(**common, H_op=copy.deepcopy(coordinates), H_tq=copy.deepcopy(coordinates), n_H=286)
    policy = dict(**copy.deepcopy(common), n_H_op=286)
    return hn, policy


def encoded(hn, policy):
    return json.dumps(hn).encode(), json.dumps(policy).encode()


def verify(hn, policy):
    h, p = encoded(hn, policy)
    return REF.verify_documents(h, p, expected_hn_sha256=hashlib.sha256(h).hexdigest(),
                                expected_policy_sha256=hashlib.sha256(p).hexdigest())


def test_valid_binding_preserves_order_hashes_and_immutable_nested_values():
    hn, policy = documents()
    hn["H_op"].reverse(); hn["H_tq"].reverse()
    result = verify(hn, policy)
    assert result.coordinates == tuple(map(tuple, hn["H_op"]))
    assert result.hn_sha256 == hashlib.sha256(encoded(hn, policy)[0]).hexdigest()
    assert result.policy_sha256 == hashlib.sha256(encoded(hn, policy)[1]).hexdigest()
    assert result.model.repo == REF.MODEL_REPO and result.model.revision == REF.MODEL_REVISION
    assert result.model.sha256 == tuple(sorted(hn["provenance"]["models"][REF.MODEL_ID]["sha256"].items()))
    with pytest.raises(FrozenInstanceError):
        result.hn_sha256 = "0"*64
    with pytest.raises(TypeError):
        result.coordinates[0][0] = 99
    with pytest.raises(FrozenInstanceError):
        result.model.repo = "other"
    output=result.to_dict(); output["coordinates"][0][0]=99; output["model"]["sha256"].clear()
    assert result.coordinates[0][0] != 99
    assert result.model.sha256


def test_digest_checks_original_bytes_before_any_json_semantics():
    hn, policy=documents(); h,p=encoded(hn,policy)
    for changed_h,changed_p in [(h+b"\n",p),(h,p+b" ")]:
        with pytest.raises(REF.ReferenceError,match="SHA-256 mismatch"):
            REF.verify_documents(changed_h,changed_p,expected_hn_sha256=hashlib.sha256(h).hexdigest(),
                                 expected_policy_sha256=hashlib.sha256(p).hexdigest())


@pytest.mark.parametrize("bad", [True, 1.0, "1", -1, 28])
def test_layer_bounds_and_integer_types_are_strict(bad):
    hn,policy=documents();hn["H_op"][0][0]=bad;hn["H_tq"][0][0]=bad
    with pytest.raises(REF.ReferenceError):verify(hn,policy)


@pytest.mark.parametrize("bad", [False, 1.5, "1", -1, 6144])
def test_channel_bounds_and_integer_types_are_strict(bad):
    hn,policy=documents();hn["H_op"][0][1]=bad;hn["H_tq"][0][1]=bad
    with pytest.raises(REF.ReferenceError):verify(hn,policy)


@pytest.mark.parametrize("change", ["duplicate", "missing", "different_order", "missing_H_tq"])
def test_selection_must_be_286_unique_identical_coordinates(change):
    hn,policy=documents()
    if change=="duplicate":hn["H_op"][1]=hn["H_op"][0];hn["H_tq"][1]=hn["H_tq"][0]
    elif change=="missing":hn["H_op"].pop();hn["H_tq"].pop()
    elif change=="different_order":hn["H_tq"].reverse()
    else:hn.pop("H_tq")
    with pytest.raises(REF.ReferenceError):verify(hn,policy)


@pytest.mark.parametrize("document", ["hn", "policy"])
@pytest.mark.parametrize("field,bad", [("h_policy","original"),("post_hoc",False),("post_hoc",1),
    ("H_op_name","H+"),("amendment",1),("amendment",True)])
def test_policy_context_must_agree_in_both_documents(document,field,bad):
    hn,policy=documents();(hn if document=="hn" else policy)[field]=bad
    with pytest.raises(REF.ReferenceError):verify(hn,policy)


@pytest.mark.parametrize("field,bad", [("n_H",285),("n_H",286.0),("n_H_op",285),("n_H_op",True)])
def test_declared_counts_are_strict(field,bad):
    hn,policy=documents();(hn if field=="n_H" else policy)[field]=bad
    with pytest.raises(REF.ReferenceError):verify(hn,policy)


@pytest.mark.parametrize("document", ["hn", "policy"])
@pytest.mark.parametrize("field,bad", [("repo","Qwen/Qwen3.6"),("revision","main"),
    ("revision","0"*40),("sha256",{}),("sha256",{"x":"not-a-digest"})])
def test_model_pins_reject_cross_model_or_unpinned_inputs(document,field,bad):
    hn,policy=documents();target=(hn if document=="hn" else policy)
    target["provenance"]["models"][REF.MODEL_ID][field]=bad
    with pytest.raises(REF.ReferenceError):verify(hn,policy)


def test_model_file_hashes_must_agree_between_documents():
    hn,policy=documents();policy["provenance"]["models"][REF.MODEL_ID]["sha256"]["synthetic-model.safetensors"]="c"*64
    with pytest.raises(REF.ReferenceError,match="different model-file pins"):verify(hn,policy)


def test_production_loader_has_no_synthetic_hash_override(tmp_path):
    hn,policy=documents();h,p=encoded(hn,policy)
    hp,pp=tmp_path/"hn_set.json",tmp_path/"htq_policy.json";hp.write_bytes(h);pp.write_bytes(p)
    with pytest.raises(REF.ReferenceError,match="SHA-256 mismatch"):
        REF.load_reference(hp,pp)


def test_binding_json_roundtrip_is_explicitly_posthoc_and_does_not_change_inputs(tmp_path):
    hn,policy=documents();result=verify(hn,policy); before=encoded(hn,policy)
    out=tmp_path/"private"/"reference.json";REF.write_reference(result,out)
    saved=json.loads(out.read_text())
    assert saved["source_sha256"]=={"hn_set.json":result.hn_sha256,"htq_policy.json":result.policy_sha256}
    assert saved["coordinates"]==hn["H_op"]
    assert saved["post_hoc"] is True and saved["confirmatory_eligible"] is False
    assert saved["model"]["repo"]==REF.MODEL_REPO
    assert encoded(hn,policy)==before
    assert not list(out.parent.glob("*.tmp"))


def test_cli_rejects_untrusted_files_without_creating_output(tmp_path):
    hn,policy=documents();h,p=encoded(hn,policy)
    hp,pp,out=tmp_path/"hn_set.json",tmp_path/"policy.json",tmp_path/"reference.json"
    hp.write_bytes(h);pp.write_bytes(p)
    with pytest.raises(SystemExit) as error:
        REF.main(["--hn-set",str(hp),"--policy",str(pp),"--out",str(out)])
    assert error.value.code==2 and not out.exists()
    assert hp.read_bytes()==h and pp.read_bytes()==p
    with pytest.raises(SystemExit):
        REF.main(["--hn-set",str(hp),"--policy",str(pp),"--out",str(hp)])
    assert hp.read_bytes()==h


def test_import_does_not_load_model_or_numeric_runtime():
    root=Path(__file__).resolve().parents[1]
    code="import sys; import attention_probe.reference; assert not {'torch','transformers','numpy'} & set(sys.modules)"
    subprocess.run([sys.executable,"-c",code],cwd=root,check=True)
