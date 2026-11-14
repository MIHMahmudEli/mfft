"""
Validation test suite for MFFT results registry, individual record provenance,
README metrics traceability, multi-level provenance hash semantics, partition
membership integrity, and DOI-readiness compliance.
"""

import json
import hashlib
import os
import sys
import yaml
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from huggingface_hub import hf_hub_download

def get_token():
    try:
        from scripts.list_active_runs import get_hf_token
        return get_hf_token()
    except Exception:
        return os.environ.get("HF_TOKEN")

def test_registry_structure():
    reg_path = Path("results/canonical_results_registry.json")
    assert reg_path.exists(), "canonical_results_registry.json must exist"
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "provenance" in data
    assert "metric_conventions" in data
    assert "provenance_hash_semantics" in data["metric_conventions"]
    sem = data["metric_conventions"]["provenance_hash_semantics"]
    assert "master_manifest_sha256" in sem
    assert "split_config_sha256" in sem
    assert "split_manifest_sha256" in sem
    assert "partition_membership_hashes" in sem
    assert sem["master_manifest_sha256"].get("scope") == "master_dataset_inventory"
    assert sem["split_config_sha256"].get("scope") == "split_hyperparameters"
    assert sem["split_manifest_sha256"].get("scope") == "legacy_compatibility"
    assert sem["partition_membership_hashes"].get("scope") == "exact_sample_membership"
    assert "experiments" in data
    assert len(data["experiments"]) == 32, f"Expected 32 experiments, got {len(data['experiments'])}"
    print("[PASS] Results registry structure, provenance hash semantics, and experiment count (32/32) verified.")

def test_individual_record_provenance_and_values():
    reg_path = Path("results/canonical_results_registry.json")
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tok = get_token()
    legacy_manifest_anchor = "e8db1a7249ef22fbf9fcb6e6659e9ecc5ebbc5d81430a268fce26546f23bdde2"

    # Independently compute SHA-256 from exact raw file bytes of master manifest artifact
    manifest_local_path = None
    for cand in [Path("dataset/split_manifest.json"), Path("split_manifest.json")]:
        if cand.is_file():
            manifest_local_path = cand
            break
    if manifest_local_path is None:
        downloaded = hf_hub_download(
            repo_id="MohsinElis/mfft-master-manifest",
            filename="manifest/split_manifest.json",
            repo_type="model",
            token=tok
        )
        manifest_local_path = Path(downloaded)

    h_raw = hashlib.sha256()
    with open(manifest_local_path, "rb") as f_raw:
        while chunk := f_raw.read(65536):
            h_raw.update(chunk)
    raw_manifest_hash = h_raw.hexdigest()

    # Verify top-level provenance raw digest
    assert data.get("provenance", {}).get("master_manifest_sha256") == raw_manifest_hash, (
        f"Registry provenance master_manifest_sha256 mismatch: {data.get('provenance', {}).get('master_manifest_sha256')} vs {raw_manifest_hash}"
    )
    assert data["metric_conventions"]["provenance_hash_semantics"]["master_manifest_sha256"]["value"] == raw_manifest_hash, (
        "Semantics master_manifest_sha256 value mismatch"
    )

    logo_expected_sizes = {
        "logo_holdout_sdxl": 9606,
        "logo_holdout_diffusion": 9606,
        "logo_holdout_gan": 12594,
        "logo_holdout_deepfake": 10743
    }

    # Verify unverified membership run set (exactly 6 runs without partition artifacts)
    expected_unverified = {
        "mfft_base",
        "mfft_large",
        "mfft_tiny",
        "ablation_band_2bands",
        "baseline_clip",
        "baseline_efficientnet_b0",
    }
    actual_unverified = {
        exp_id for exp_id, exp_data in data["experiments"].items()
        if not exp_data.get("provenance_metadata", {}).get("membership_verified", False)
    }
    assert actual_unverified == expected_unverified, (
        f"Mismatch in unverified membership runs: {actual_unverified} vs {expected_unverified}"
    )

    for exp_id, exp_data in data["experiments"].items():
        pm = exp_data.get("provenance_metadata", {})
        cfg = exp_data.get("config", {})
        m = exp_data.get("metrics", {})

        # 1. Master manifest raw digest & legacy compatibility alias
        assert pm.get("master_manifest_sha256") == raw_manifest_hash, (
            f"{exp_id}: invalid raw master manifest hash {pm.get('master_manifest_sha256')} vs {raw_manifest_hash}"
        )
        assert pm.get("split_manifest_sha256") == legacy_manifest_anchor, (
            f"{exp_id}: invalid legacy split manifest hash {pm.get('split_manifest_sha256')}"
        )
        assert pm.get("checkpoint_path_hf") == f"runs/{exp_id}/checkpoints/best.pt", f"{exp_id}: invalid checkpoint path"
        expected_metric_path = (
            f"runs/{exp_id}/tables/table1_main_performance.csv"
            if exp_id == "ablation_spatial_only"
            else f"runs/{exp_id}/metrics/test_metrics.json"
        )
        assert pm.get("test_metrics_path_hf") == expected_metric_path, f"{exp_id}: invalid metrics path"
        assert pm.get("training_script") == "kaggle_mfft_pipeline.py", f"{exp_id}: invalid training script"
        assert pm.get("source_repo") == "MohsinElis/mfft-checkpoints", f"{exp_id}: invalid source repo"
        assert pm.get("verification_status") == "VERIFIED_GPU_EVALUATION", f"{exp_id}: unverified status"

        # 2. Seed validation & independent recomputation of split_config_sha256
        expected_seed = 42
        if "seed999" in exp_id:
            expected_seed = 999
        elif "seed1234" in exp_id:
            expected_seed = 1234
        elif "seed2026" in exp_id:
            expected_seed = 2026
        assert pm.get("seed") == expected_seed, f"{exp_id}: expected seed {expected_seed}, got {pm.get('seed')}"
        assert cfg.get("seed") == expected_seed, f"{exp_id}: config seed mismatch"

        expected_config_payload = {
            "seed": expected_seed,
            "test": 18000,
            "total": 120000,
            "train": 84000,
            "val": 18000
        }
        recomputed_config_hash = hashlib.sha256(json.dumps(expected_config_payload, sort_keys=True).encode("utf-8")).hexdigest()
        assert pm.get("split_config_sha256") == recomputed_config_hash, f"{exp_id}: split_config_sha256 mismatch"

        # 3. Membership verification flag & field consistency
        if pm.get("membership_verified"):
            assert pm.get("test_indices_sha256") is not None and len(pm.get("test_indices_sha256")) == 64
            assert pm.get("train_indices_sha256") is not None and len(pm.get("train_indices_sha256")) == 64
            assert pm.get("val_indices_sha256") is not None and len(pm.get("val_indices_sha256")) == 64
            assert pm.get("split_indices_path_hf") == f"runs/{exp_id}/configs/split_indices.json"
        else:
            assert pm.get("test_indices_sha256") is None
            assert pm.get("train_indices_sha256") is None
            assert pm.get("val_indices_sha256") is None
            assert pm.get("split_indices_path_hf") is None

        # 4. Model variant validation
        variant = pm.get("model_variant")
        assert variant is not None and len(variant) > 0, f"{exp_id}: missing model_variant in provenance"

        # 5. Test split size validation
        actual_size = exp_data.get("test_split_size")
        if exp_id in logo_expected_sizes:
            assert actual_size == logo_expected_sizes[exp_id], f"{exp_id}: expected size {logo_expected_sizes[exp_id]}, got {actual_size}"
        else:
            assert actual_size == 18000, f"{exp_id}: expected size 18000, got {actual_size}"

        # 6. Metrics sanity validation
        acc = m.get("accuracy")
        auc = m.get("auc", m.get("roc_auc"))
        loss = m.get("loss")
        assert 0.0 <= acc <= 100.0, f"{exp_id}: accuracy out of range {acc}"
        assert 0.0 <= auc <= 100.0, f"{exp_id}: auc out of range {auc}"
        if loss is not None:
            assert loss > 0.0, f"{exp_id}: invalid loss {loss}"

    # 7. Specific check on resolved GAP ablation
    gap_metrics = data["experiments"]["ablation_fusion_avg"]["metrics"]
    assert abs(gap_metrics["accuracy"] - 98.9167) < 0.01, f"GAP accuracy mismatch: {gap_metrics['accuracy']}"
    assert abs(gap_metrics["auc"] - 99.9329) < 0.01, f"GAP AUC mismatch: {gap_metrics['auc']}"

    # 8. Specific check on distinct architectures (MFFT-Large vs 4-band Base)
    assert data["metric_conventions"]["discrepancy_resolutions"]["parameter_count_architecture_distinction"]["mfft_large"]["parameters"] == 6301250
    assert data["metric_conventions"]["discrepancy_resolutions"]["parameter_count_architecture_distinction"]["ablation_band_4bands"]["parameters"] == 1917218

    print("[PASS] Individual record provenance, split configuration hashes, and values verified across all 32 experiments.")

def test_partition_membership_integrity():
    reg_path = Path("results/canonical_results_registry.json")
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    tok = get_token()
    tested_count = 0
    
    # Representative target runs spanning canonical base, varied seeds, large, and ablations
    verification_targets = [
        "mfft_base_seed42",
        "mfft_base_seed999",
        "mfft_base_seed1234",
        "mfft_base_seed2026",
        "mfft_large_seed42",
        "ablation_band_4bands",
        "baseline_resnet50",
    ]
    
    for exp_id in verification_targets:
        exp_data = data["experiments"][exp_id]
        pm = exp_data["provenance_metadata"]
        assert pm.get("membership_verified"), f"{exp_id} should have verified membership"
        
        idx_path = pm["split_indices_path_hf"]
        local_path = hf_hub_download(
            repo_id="MohsinElis/mfft-checkpoints",
            filename=idx_path,
            repo_type="model",
            token=tok
        )
        with open(local_path, "r", encoding="utf-8") as f:
            idx_d = json.load(f)
            
        train_idx = idx_d["train_indices"]
        val_idx = idx_d["val_indices"]
        test_idx = idx_d["test_indices"]
        
        # 1. Exact partition counts
        assert len(train_idx) == 84000, f"{exp_id}: train count {len(train_idx)}"
        assert len(val_idx) == 18000, f"{exp_id}: val count {len(val_idx)}"
        assert len(test_idx) == 18000, f"{exp_id}: test count {len(test_idx)}"
        
        # 2. No duplicate indices within partitions
        assert len(set(train_idx)) == 84000, f"{exp_id}: duplicate train indices"
        assert len(set(val_idx)) == 18000, f"{exp_id}: duplicate val indices"
        assert len(set(test_idx)) == 18000, f"{exp_id}: duplicate test indices"
        
        # 3. Mutually disjoint partitions
        s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
        assert len(s_train & s_val) == 0, f"{exp_id}: train and val overlap"
        assert len(s_train & s_test) == 0, f"{exp_id}: train and test overlap"
        assert len(s_val & s_test) == 0, f"{exp_id}: val and test overlap"
        assert len(s_train | s_val | s_test) == 120000, f"{exp_id}: partitions do not cover 120,000 images"
        
        # 4. In range [0, 120000)
        assert min(min(train_idx), min(val_idx), min(test_idx)) >= 0, f"{exp_id}: negative index"
        assert max(max(train_idx), max(val_idx), max(test_idx)) < 120000, f"{exp_id}: index out of bounds"
        
        # 5. Cryptographic hash reproduction (compact JSON serialization)
        recomputed_train_hash = hashlib.sha256(json.dumps(train_idx, separators=(",", ":")).encode("utf-8")).hexdigest()
        recomputed_val_hash = hashlib.sha256(json.dumps(val_idx, separators=(",", ":")).encode("utf-8")).hexdigest()
        recomputed_test_hash = hashlib.sha256(json.dumps(test_idx, separators=(",", ":")).encode("utf-8")).hexdigest()
        
        assert pm["train_indices_sha256"] == recomputed_train_hash, f"{exp_id}: train_indices_sha256 mismatch"
        assert pm["val_indices_sha256"] == recomputed_val_hash, f"{exp_id}: val_indices_sha256 mismatch"
        assert pm["test_indices_sha256"] == recomputed_test_hash, f"{exp_id}: test_indices_sha256 mismatch"
        
        tested_count += 1
        
    print(f"[PASS] Partition membership integrity verified across {tested_count} targeted runs (disjointness, counts, bounds, and SHA-256 digests reproduced).")

def test_readme_traceability():
    reg_path = Path("results/canonical_results_registry.json")
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    
    readme_path = Path("README.md")
    assert readme_path.exists()
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    
    key_checks = {
        "mfft_tiny": ("98.59", "99.88"),
        "mfft_base": ("98.37", "99.87"),
        "mfft_large": ("98.11", "99.68"),
        "baseline_efficientnet_b0": ("99.51", "99.95"),
        "baseline_resnet50": ("99.38", "99.96"),
        "baseline_vit_b16": ("93.13", "98.37"),
        "baseline_clip": ("85.76", "92.66"),
        "baseline_freqdetect": ("50.00", "49.95")
    }
    
    for exp_id, (acc_str, auc_str) in key_checks.items():
        assert acc_str in readme, f"README missing accuracy {acc_str} for {exp_id}"
        assert auc_str in readme, f"README missing AUC {auc_str} for {exp_id}"
        exp_m = reg["experiments"][exp_id]["metrics"]
        assert abs(exp_m["accuracy"] - float(acc_str)) < 0.05
        assert abs(exp_m.get("auc", exp_m.get("roc_auc")) - float(auc_str)) < 0.05
    
    print("[PASS] All key README metrics verified against canonical registry.")

def test_multiseed_reproducibility():
    reg_path = Path("results/canonical_results_registry.json")
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    
    seeds = ["mfft_base", "mfft_base_seed42", "mfft_base_seed999", "mfft_base_seed1234", "mfft_base_seed2026"]
    accs = [reg["experiments"][s]["metrics"]["accuracy"] for s in seeds]
    aucs = [reg["experiments"][s]["metrics"].get("auc", reg["experiments"][s]["metrics"].get("roc_auc")) for s in seeds]
    
    mean_acc = np.mean(accs)
    std_acc_ddof0 = np.std(accs, ddof=0)
    std_acc_ddof1 = np.std(accs, ddof=1)
    
    mean_auc = np.mean(aucs)
    std_auc_ddof0 = np.std(aucs, ddof=0)
    std_auc_ddof1 = np.std(aucs, ddof=1)
    
    assert abs(mean_acc - 98.5833) < 0.01, f"Mean acc {mean_acc} mismatch"
    assert abs(std_acc_ddof0 - 0.1294) < 0.01, f"Std acc ddof0 {std_acc_ddof0} mismatch"
    assert abs(std_acc_ddof1 - 0.1447) < 0.01, f"Std acc ddof1 {std_acc_ddof1} mismatch"
    
    assert abs(mean_auc - 99.8920) < 0.01, f"Mean auc {mean_auc} mismatch"
    assert abs(std_auc_ddof0 - 0.0183) < 0.01, f"Std auc ddof0 {std_auc_ddof0} mismatch"
    assert abs(std_auc_ddof1 - 0.0204) < 0.01, f"Std auc ddof1 {std_auc_ddof1} mismatch"
    
    print(f"[PASS] 5-seed statistics reproduced: Acc={mean_acc:.2f}% (std0={std_acc_ddof0:.2f}%, std1={std_acc_ddof1:.2f}%), AUC={mean_auc:.2f}% (std0={std_auc_ddof0:.2f}%, std1={std_auc_ddof1:.2f}%).")

def test_zenodo_and_citation_metadata():
    with open(".zenodo.json", "r", encoding="utf-8") as f:
        zen = json.load(f)
    assert zen["title"] == "MFFT: Multi-Frequency Fusion Transformer for Explainable AI-Generated Image Detection"
    assert len(zen["creators"]) == 3
    assert zen["license"] == "MIT"
    
    with open("CITATION.cff", "r", encoding="utf-8") as f:
        cff = yaml.safe_load(f)
    assert cff["doi"] in ["10.5281/zenodo.23339178", "10.5281/zenodo.23484098"]
    assert cff["version"] in ["1.0.0", "1.0.1"]
    assert cff["repository-code"] == "https://github.com/MIHMahmudEli/mfft"
    print("[PASS] Metadata in .zenodo.json and CITATION.cff strictly validated.")

if __name__ == "__main__":
    test_registry_structure()
    test_individual_record_provenance_and_values()
    test_partition_membership_integrity()
    test_readme_traceability()
    test_multiseed_reproducibility()
    test_zenodo_and_citation_metadata()
    print("\nALL RESULTS REGISTRY & METADATA TESTS PASSED SUCCESSFULLY!")
