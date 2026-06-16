import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from collections import OrderedDict
from sklearn.metrics import f1_score
from tabulate import tabulate

from .build_evaluator import EVALUATOR_REGISTRY
from metrics.euclidean_dist import euclidean_dist
from metrics.mAP_cmc import mAP_cmc


@EVALUATOR_REGISTRY.register()
class R1_mAP:
    def __init__(self, cfg, num_query, max_rank = 50, reranking = False):
        self.cfg = cfg
        self.num_query = num_query
        self.max_rank = max_rank
        self.reranking = reranking

        self.feats = []
        self.aids = []
        self.camids = []
        self.domains = []
        self.img_paths = []
    
    def reset(self):
        self.feats = []
        self.aids = []
        self.camids = []
        self.domains = []
        self.img_paths = []
        
    def process(self, batch_output):
        if len(batch_output) == 4:
            feats, aids, camids, domains = batch_output
            img_paths = None
        else:
            feats, aids, camids, domains, img_paths = batch_output
        self.feats.append(feats)
        self.aids.extend(aids)
        self.camids.extend(camids)
        self.domains.extend(domains)
        if img_paths is not None:
            self.img_paths.extend(img_paths)

    def evaluate(self, display_ranks = [1, 5, 10]):
        results = OrderedDict()
        features = torch.cat(self.feats, dim = 0)    # Shape: (num_query + num_gallery, output_dim)
        
        query_feats = features[:self.num_query, :]                # Shape: (num_query, output_dim)
        query_aids = np.asarray(self.aids[:self.num_query])       # Shape: (num_query,)
        query_camids = np.asarray(self.camids[:self.num_query])   # Shape: (num_query,)
        query_domains = np.asarray(self.domains[:self.num_query]) # Shape: (num_query,)

        gallery_feats = features[self.num_query:, :]              # Shape: (num_gallery, output_dim)
        gallery_aids = np.asarray(self.aids[self.num_query:])     # Shape: (num_gallery,)
        gallery_camids = np.asarray(self.camids[self.num_query:]) # Shape: (num_gallery,)
        gallery_domains = np.asarray(self.domains[self.num_query:]) # Shape: (num_gallery,)
        img_paths = np.asarray(self.img_paths) if self.img_paths else None
        if img_paths is not None and len(img_paths) == (self.num_query + len(gallery_feats)):
            query_img_paths = img_paths[:self.num_query]
            gallery_img_paths = img_paths[self.num_query:]
        else:
            query_img_paths = None
            gallery_img_paths = None

        dist_mat = np.asarray(euclidean_dist(query_feats, gallery_feats))    # Shape: (num_query, num_gallery)
        # make sure only the same domain distances are considered

        # compute per-domain metrics (by query domain)
        per_domain_results = None
        if self.domains is not None:
            unique_query_domains = np.unique(query_domains)
            per_domain_results = OrderedDict()
            for dom in unique_query_domains:
                print(f"Evaluating domain: {self.cfg.DATASET.TARGET_DOMAINS[dom]}")
                dom_query_mask = (query_domains == dom)
                dom_gallery_mask = (gallery_domains == dom)
                if not np.any(dom_query_mask):
                    continue
                # print(f"  Found {np.sum(dom_query_mask)} queries and {np.sum(dom_gallery_mask)} galleries for domain {dom}")
                # print(f"  Unique query AIDs: {np.unique(query_aids[dom_query_mask])}")
                # print(f"  Unique gallery AIDs: {np.unique(gallery_aids[dom_gallery_mask])}")
                dom_dist = dist_mat[dom_query_mask, :][:, dom_gallery_mask]
                dom_query_aids = query_aids[dom_query_mask]
                dom_gallery_aids = gallery_aids[dom_gallery_mask]
                dom_query_camids = query_camids[dom_query_mask]
                dom_gallery_camids = gallery_camids[dom_gallery_mask]

                if self.cfg.MODEL.Supress_Minimal_Distance_IDs:
                    # filter out identical images (0 distance)
                    identical_mask = np.isclose(dom_dist, 0, atol=1e-6)
                    dom_dist[identical_mask] = np.inf
                    dom_gallery_aids[identical_mask.any(axis=0)] = -1  # Invalidate identical gallery aids
                    print(f"Filtered out {np.sum(identical_mask)} identical image pairs for domain {dom}")

                    # Drop queries that no longer have a valid positive after removal
                    valid_query_indices = []
                    for q_idx, q_aid in enumerate(dom_query_aids):
                        if (dom_gallery_aids == q_aid).any():
                            valid_query_indices.append(q_idx)
                    if not valid_query_indices:
                        print(f"No valid positives remain for domain {dom} after identical filtering; skipping.")
                        per_domain_results[int(dom)] = (np.zeros(self.max_rank, dtype=float), 0.0)
                        continue
                    dom_dist = dom_dist[valid_query_indices, :]
                    dom_query_aids = dom_query_aids[valid_query_indices]
                    dom_query_camids = dom_query_camids[valid_query_indices]

                try:
                    dom_cmc, dom_mAP = mAP_cmc(dom_dist, dom_query_aids, dom_gallery_aids, dom_query_camids, dom_gallery_camids, self.max_rank)
                    per_domain_results[int(dom)] = (dom_cmc, dom_mAP)
                except (AssertionError, RuntimeError):
                    # No valid queries for this domain (no positives in gallery)
                    # Use zero CMC curve of length max_rank and zero mAP
                    per_domain_results[int(dom)] = (np.zeros(self.max_rank, dtype=float), 0.0)

        self.display(per_domain_results, display_ranks)
        self._visualize_rank_samples(
            dist_mat,
            query_aids,
            query_camids,
            query_domains,
            gallery_aids,
            gallery_camids,
            gallery_domains,
            query_img_paths,
            gallery_img_paths,
        )

        summary = OrderedDict()
        if per_domain_results is not None:
            for dom, (cmc, mAP) in per_domain_results.items():
                dom_name = self.cfg.DATASET.TARGET_DOMAINS[dom]
                summary[f"{dom_name}/mAP"] = float(mAP)
                summary[f"{dom_name}/Rank-1"] = float(cmc[0])
                summary[f"{dom_name}/Rank-5"] = float(cmc[4])
                summary[f"{dom_name}/Rank-10"] = float(cmc[9])

            if len(per_domain_results) > 1:
                summary["mAP"] = float(np.mean([res[1] for res in per_domain_results.values()]))
                summary["Rank-1"] = float(np.mean([res[0][0] for res in per_domain_results.values()]))
            elif len(per_domain_results) == 1:
                dom = list(per_domain_results.keys())[0]
                cmc, mAP = per_domain_results[dom]
                summary["mAP"] = float(mAP)
                summary["Rank-1"] = float(cmc[0])

        return summary
    
    def display(self, results, display_ranks):
        evaluation_table = []
        evaluation_table.append(["Domain", "mAP", "Rank-1", "Rank-5", "Rank-10"])
        for domain, (cmc, mAP) in results.items():
            row = [self.cfg.DATASET.TARGET_DOMAINS[domain]]
            mAP_str = f"{mAP:.2%}"
            row.append(mAP_str)
            for r in display_ranks:
                # Ensure cmc is an array and has enough elements
                rank = f"{cmc[r - 1]:.2%}"
                row.append(rank)
            evaluation_table.append(row)
        print(tabulate(evaluation_table))
    
    def _visualize_rank_samples(
        self,
        dist_mat,
        query_aids,
        query_camids,
        query_domains,
        gallery_aids,
        gallery_camids,
        gallery_domains,
        query_img_paths,
        gallery_img_paths,
    ):
        vis_cfg = getattr(self.cfg.TEST, "VISUALIZE", None)
        if vis_cfg is None or not getattr(vis_cfg, "ENABLED", False):
            return

        if query_img_paths is None or gallery_img_paths is None:
            print("Visualization skipped: image paths not available in evaluator.")
            return

        num_queries = len(query_img_paths)
        num_gallery = len(gallery_img_paths)
        if num_queries == 0 or num_gallery == 0:
            return

        topk = max(1, int(getattr(vis_cfg, "TOPK", 5)))
        sample_size = min(int(getattr(vis_cfg, "NUM_QUERIES", 10)), num_queries)
        if sample_size <= 0:
            return

        same_domain_only = bool(getattr(vis_cfg, "SAME_DOMAIN_ONLY", True))
        rng_seed = getattr(self.cfg, "SEED", None)
        rng = np.random.default_rng(rng_seed)
        if num_queries <= sample_size:
            sampled_indices = np.arange(num_queries)
        else:
            sampled_indices = rng.choice(num_queries, size=sample_size, replace=False)

        output_subdir = getattr(vis_cfg, "OUTPUT_SUBDIR", "rank_visualizations")
        output_dir = os.path.join(self.cfg.OUTPUT_DIR, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        for q_idx in sampled_indices:
            query_path = query_img_paths[q_idx]
            if not os.path.isfile(query_path):
                continue

            query_domain = query_domains[q_idx]
            query_aid = query_aids[q_idx]
            query_cam = query_camids[q_idx]

            gallery_candidates = np.arange(num_gallery)
            if same_domain_only:
                domain_mask = gallery_domains == query_domain
                gallery_candidates = gallery_candidates[domain_mask]
            if gallery_candidates.size == 0:
                continue

            candidate_dists = dist_mat[q_idx, gallery_candidates]
            sorted_indices = gallery_candidates[np.argsort(candidate_dists)]
            if sorted_indices.size == 0:
                continue

            ranked_indices = sorted_indices[: min(topk, sorted_indices.size)]
            gallery_info = []
            for rank_pos, g_idx in enumerate(ranked_indices, start=1):
                gallery_path = gallery_img_paths[g_idx]
                if not os.path.isfile(gallery_path):
                    continue
                gallery_info.append(
                    {
                        "path": gallery_path,
                        "aid": gallery_aids[g_idx],
                        "camid": gallery_camids[g_idx],
                        "domain": gallery_domains[g_idx],
                        "distance": float(dist_mat[q_idx, g_idx]),
                        "rank": rank_pos,
                        "is_match": gallery_aids[g_idx] == query_aid,
                    }
                )

            if not gallery_info:
                continue

            figure_name = f"query_{q_idx:05d}_aid{query_aid}_dom{query_domain}.png"
            output_path = os.path.join(output_dir, figure_name)
            self._save_rank_figure(
                query_path=query_path,
                query_aid=query_aid,
                query_cam=query_cam,
                query_domain=query_domain,
                gallery_info=gallery_info,
                output_path=output_path,
            )
    
    @staticmethod
    def _save_rank_figure(
        query_path,
        query_aid,
        query_cam,
        query_domain,
        gallery_info,
        output_path,
    ):
        try:
            query_img = Image.open(query_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return

        num_cols = len(gallery_info) + 1
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 8))
        if num_cols == 1:
            axes = [axes]

        axes[0].imshow(query_img)
        axes[0].axis("off")
        axes[0].set_title(
            f"query\nid:{query_aid} cam:{query_cam} dom:{query_domain}",
            fontsize=10,
        )

        for axis, info in zip(axes[1:], gallery_info):
            try:
                gallery_img = Image.open(info["path"]).convert("RGB")
            except (FileNotFoundError, OSError):
                axis.axis("off")
                axis.set_title("missing", fontsize=10)
                continue
            axis.imshow(gallery_img)
            axis.axis("off")
            status = "hit" if info["is_match"] else "miss"
            axis.set_title(
                f"{status} r{info['rank']} id:{info['aid']} cam:{info['camid']}\n"
                f"d:{info['distance']:.3f}",
                fontsize=10,
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    

@EVALUATOR_REGISTRY.register()
class Classification:
    def __init__(self, cfg, class_label_name_mapping = None):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def process(self, model_output, ground_truth):
        pred = model_output.max(1)[1]
        matches = pred.eq(ground_truth).float()
        self._correct += int(matches.sum().item())
        self._total += ground_truth.shape[0]
        self._y_true.extend(ground_truth.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        accuracy = 100.0 * self._correct / self._total
        error_rate = 100.0 - accuracy
        macro_f1 = 100.0 * f1_score(
            self._y_true, self._y_pred, average = "macro", labels = np.unique(self._y_true)
        )

        results["accuracy"] = accuracy
        results["error_rate"] = error_rate
        results["macro_f1"] = macro_f1

        evaluation_table = [
            ["Total #", f"{self._total:,}"], 
            ["Correct #", f"{self._correct:,}"], 
            ["Accuracy", f"{accuracy:.2f}%"], 
            ["Error Rate", f"{error_rate:.2f}%"], 
            ["Macro_F1", f"{macro_f1:.2f}%"],
        ]
        print(tabulate(evaluation_table))

        return results

