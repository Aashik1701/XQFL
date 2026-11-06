"""Promote verified release results into the current Markdown manuscript."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.current_rq4 import mia_table, noise_table

MIA = mia_table()
NOISE = noise_table()
LABEL = {'angle':'Angle', 'amplitude':'Amplitude', 'reuploading':'Re-uploading', 'iqp':'IQP'}
ORDER = list(LABEL)
CHANGES = []

METHOD = r'''For RQ4, each explained input has membership label $y_{\mathrm{mem}}=1$ for a training record and $0$ for a held-out test record. Each target supplies 50 member and 50 non-member four-component Q-SHAP vectors. Both groups use the same 20-record training-derived background, disjoint from the queried member records, and the same eight-permutation estimator. This background is not an independent public reference. Standardization and logistic regression are fitted only within each outer training fold. The primary statistic is the mean of five stratified held-out fold AUCs:

$$\mathrm{AUC}_{\mathrm{MIA}}=\frac{1}{5}\sum_{h=1}^{5}\operatorname{AUC}\!\left(\{(y_{\mathrm{mem}}^{(j)},g_{-h}(\hat{\boldsymbol\phi}(x_j))) : j\in I_h\}\right). \tag{11}$$

Here $I_h$ is outer test fold $h$ and $g_{-h}$ is the attack pipeline trained without that fold. AUC 0.5 denotes chance-level ranking and 1 denotes perfect ranking, not an attack-success probability. A below-chance AUC is not evidence of privacy: reversing score direction can reveal discrimination. We additionally retain $\max(\mathrm{AUC},1-\mathrm{AUC})$ as a post-hoc orientation diagnostic, not a validated attack result. A separate direction-aware evaluation chooses score orientation and the TPR-minus-FPR threshold using four inner folds of the outer training data only. Held-out scores, labels, fold assignments and attack advantage are retained. Pooled out-of-fold TPR at low FPR is descriptive; with 50 non-members its empirical FPR resolution is 0.02. Direct MIA remains a signal-presence probe for this attack family, not a realistic attacker-success estimate or a privacy certificate. Historical shadow-model results were not re-evaluated and are excluded from current comparisons.

### 6.5b Calibrated Laplace-Noise Evaluation

Each complete set of 100 released vectors receives one independently sampled noise matrix, without reusing member noise for non-members. For a fixed target model, we evaluate the clean release and the tested noise levels $\epsilon\in\{0.1,0.5,1,5\}$:

$$\tilde\phi_i=\phi_i+\eta_i,\qquad \eta_i\stackrel{\mathrm{iid}}{\sim}\mathrm{Laplace}\!\left(0,\frac{\Delta f_{\mathrm{emp}}(\mathrm{enc})}{\epsilon}\right),\quad i=1,\ldots,4,\qquad \Delta f_{\mathrm{emp}}(\mathrm{enc})=\frac{1}{|\mathcal R_{\mathrm{enc}}|}\sum_{(s,t)\in\mathcal R_{\mathrm{enc}}}\|\boldsymbol\phi_{g,s}^{(t)}\|_2. \tag{12}$$

The calibration set $\mathcal R_{\mathrm{enc}}$ contains the corrected IID Track-A global attribution logs for rounds $t\geq1$. The empirical scales are 0.229 (Angle), 0.262 (Amplitude), 0.344 (Re-uploading) and 0.287 (IQP). These are observed attribution scales, **not certified global sensitivity bounds**. Accordingly, $\epsilon$ indexes noise strength; **no formal $(\epsilon,\delta)$-DP guarantee is claimed**. Utility is the mean cosine similarity between each clean per-sample vector and its actual noisy release used by the attack, summarized across target seeds; undefined zero-norm cosines are counted rather than imputed.

An audit of the earlier release evaluation identified reused noise streams across the member/non-member groups, pooled standardization before cross-validation, and different explanation backgrounds. The corrected protocol regenerates all 115 RQ4 targets and evaluates 195 releases (115 clean and 80 noisy), with partition-integrity checks throughout. This correction replaces the earlier privacy results, including the interpretation of below-chance AUC as mitigation. It does not change the RQ1–RQ3 experiments. Because target retraining and evaluation corrections were combined, their individual contributions to the changed AUCs cannot be isolated.'''

CAP6 = 'Corrected direct membership-inference evaluation. (A) Mean ± one SD across 15 target seeds for each Wisconsin encoding and the Pima Amplitude check. (B) Per-seed means of five held-out fold AUCs. The dashed line is chance (0.5); no clean IID mean is resolved above chance under the study\'s two-SE convention. All IID targets are included: Wisconsin Angle and Pima Amplitude have 14/15 Track-A targets; the others have 15/15. Track-A-only sensitivity gives the same qualitative conclusion. No historical shadow-model scores are mixed with these results. AUC measures a tested discrimination signal, not realistic attacker success or a privacy guarantee.'
CAP7 = 'Corrected calibrated Laplace-noise evaluation on Wisconsin IID, five Track-A target seeds per encoding and noise level. (A) Raw direct-MIA AUC against the four categorical noise-level indices. (B) The same AUCs against cosine similarity of the actual clean/noisy per-sample releases. Lines join tested settings only; error bars show ± one SD across seeds on each plotted axis. The dashed line is chance (0.5). Clean baselines already have no resolved IID signal; the sweep does not establish monotonic leakage suppression or an effective privacy budget. The isolated IQP result at epsilon = 1 exceeds chance under an unadjusted two-SE comparison and remains exploratory. Empirical calibration is not certified sensitivity, and no formal privacy guarantee is claimed.'

def table4():
    lines = ['**Table 4. Corrected clean direct-MIA AUC, mean ± SD across target seeds. IID includes all targets; Track-A-only sensitivity is reported in the text.**', '',
             '| Dataset | Encoding | n | Direct MIA AUC | Track A | Above chance at 2 SE? |', '|---|---|---:|---:|---:|---|']
    for dataset, enc in [('wisconsin',e) for e in ORDER] + [('pima','amplitude')]:
        r = MIA.query('dataset == @dataset and encoding == @enc and distribution == "iid"').iloc[0]
        lines.append(f'| {dataset.title()} | {LABEL[enc]} | {r.n} | {r.auc_mean:.3f} ± {r.auc_std:.3f} | {r.n_track_a}/{r.n_total} | No |')
    return '\n'.join(lines)

def table5():
    lines = ['**Table 5. Corrected calibrated Laplace-noise sweep: direct-MIA AUC, mean ± SD across five Track-A targets per cell. Epsilon is a noise-level index, not a certified privacy budget.**', '',
             '| ε | Angle | Amplitude | Re-uploading | IQP |', '|---:|---:|---:|---:|---:|']
    for eps in [.1,.5,1.,5.]:
        cells=[]
        for enc in ORDER:
            r=NOISE.query('encoding == @enc and epsilon == @eps').iloc[0]
            cells.append(f'{r.auc_mean:.3f} ± {r.auc_std:.3f}')
        lines.append(f'| {eps:g} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)

RESULT = '''### 7.4 RQ4: Corrected Membership-Inference and Release-Noise Evaluation

The corrected clean IID evaluation does not reproduce the earlier encoding-dependent membership-leakage ranking. Wisconsin mean direct-MIA AUCs are 0.489 ± 0.078 (Angle), 0.487 ± 0.068 (Amplitude), 0.479 ± 0.085 (Re-uploading), and 0.504 ± 0.059 (IQP), with 15 target seeds per encoding. Pima Amplitude gives 0.501 ± 0.078 at 15 seeds (Table 4, Figure 6). No IID mean is resolved above chance under the study's two-SE convention, and no between-encoding IID difference is resolved. This is an absence of detected discrimination by the evaluated attack, not evidence that explanations contain no membership information.

The IID summary retains all 15 targets, as in the original design. Wisconsin Angle and Pima Amplitude each have 14 functional targets; Track-A-only AUCs are respectively 0.490 ± 0.081 and 0.504 ± 0.080. The other IID groups contain 15/15 Track-A targets. This sensitivity analysis does not alter the qualitative conclusion. Direction selection using inner training folds also does not establish IID discrimination. Post-hoc score flipping is reported only as a diagnostic and cannot be used to select an attack on the test labels.

{table4}

The corrected noise sweep holds each target fixed across release settings and measures utility on exactly the vectors scored by the attack (Table 5, Figure 7). It does not reproduce monotonic leakage suppression. At epsilon = 0.1, mean raw AUCs span 0.463–0.512, rather than the earlier below-chance values. One noisy cell, IQP at epsilon = 1, has AUC 0.561 ± 0.051 and exceeds chance by approximately 2.65 SE; this is an isolated, unadjusted comparison among the tested cells, not a stable encoding ranking or evidence for a mitigation mechanism. Neither a universal effective epsilon nor encoding-specific required privacy budgets are established.

{table5}

Stronger noise degrades explanation utility: mean matched cosine similarities at epsilon = 0.1 span 0.013–0.063, compared with 0.701–0.829 at epsilon = 5. These values describe retained attribution direction under the tested releases, not a certified privacy–utility frontier. The clean attack is already near chance, so a reduction in a noisy point estimate cannot establish successful suppression of a previously demonstrated membership signal.

For Wisconsin non-IID conditions, Track-A quantity-skew mean AUCs are 0.463 (Angle), 0.486 (Amplitude), 0.520 (Re-uploading), and 0.528 (IQP), each based on five targets. At alpha = 0.1, Amplitude has 0.458 ± 0.074 (5/5 Track A), Re-uploading 0.496 ± 0.050 (4/5), and IQP 0.467 ± 0.063 (5/5); Angle has only 1/5 Track-A targets and is insufficient for the primary summary. These corrected results do not support the earlier quantity-skew leakage leadership or encoding-specific leakage claims. Unequal functional coverage continues to limit severe-skew comparisons. Pima non-IID attacks and Pima noise sweeps remain untested.

Attack advantage and direction-aware AUC, together with post-hoc orientation and low-FPR diagnostics, are provided in the reproducible RQ4 diagnostic output. Negative advantage estimates are retained rather than clipped. With only 50 non-members per target, the pooled empirical FPR changes in increments of 0.02; a TPR at FPR at most 0.01 is a descriptive zero-false-positive operating point, not a validated one-percent-FPR guarantee. The corrected conclusions are conditional on this sample size, attack family and training-derived reference set.

![{cap6}](results/outputs/final/fig06_membership.pdf)

![{cap7}](results/outputs/final/fig07_privacy_utility.pdf)

'''

def main():
    p=ROOT/'XQFL_paper.md'
    text=p.read_text()
    def paragraph(prefix, new):
        nonlocal text
        blocks=text.split('\n\n')
        ids=[i for i,b in enumerate(blocks) if b.startswith(prefix)]
        if len(ids)!=1: raise ValueError((prefix,ids))
        i=ids[0]; CHANGES.append({'before':blocks[i], 'after':new}); blocks[i]=new
        text='\n\n'.join(blocks)
    # Preserve all non-RQ4 wording and numbers in the abstract/conclusion.
    old=next(b for b in text.split('\n\n') if b.startswith('Federated learning enables'))
    a=old.index('Direct membership-inference AUC'); b=old.index('On the harder dataset',a)
    paragraph('Federated learning enables',old[:a]+'The corrected direct membership-inference evaluation detects no resolved clean IID signal across encodings; the calibrated Laplace sweep reduces attribution utility but does not establish effective privacy mitigation. These attack-specific findings do not certify privacy. '+old[b:])
    paragraph('**RQ4 (Privacy).** Can the released explanation','**RQ4 (Privacy).** Can released explanation vectors discriminate training members from non-members under the evaluated attack, and how does empirically calibrated release noise affect that signal and explanation utility?')
    paragraph('**RQ4 (Privacy).** Can the released attribution','**RQ4 (Privacy).** Can released attribution vectors discriminate training-set membership under the evaluated attack, and how does calibrated noise affect that signal and utility? Formalized in Eqs. (11)–(12).')
    paragraph('4. **A privacy characterization.**','4. **A corrected privacy evaluation.** A fold-isolated direct-MIA protocol with a common explanation background and independently sampled release noise evaluates clean membership signal and matched noisy-explanation utility. Across 15 IID targets per group, the corrected analysis does not establish membership discrimination or a cross-encoding leakage ranking. The release sweep quantifies utility degradation but does not establish effective mitigation or a formal privacy guarantee.')
    old=next(b for b in text.split('\n\n') if b.startswith('ESQFL [7] is'))
    paragraph('ESQFL [7] is',old.replace('we measure, per encoding, how much privacy budget is actually required to suppress an explanation-based membership attack.','we evaluate, per encoding, empirical membership discrimination and its response to calibrated explanation-release noise, without claiming certified differential privacy.'))
    a=text.index('For RQ4, each explained input'); b=text.index('### 6.6 Statistical Protocol',a)
    CHANGES.append({'before':text[a:b].strip(),'after':METHOD})
    text=text[:a]+METHOD+'\n\n'+text[b:]
    text=text.replace('and every seed now enters every condition mean.','and no seed is excluded as anomalous. RQ1–RQ3 retain their functional-model filtering. RQ4 IID summaries include all targets with a Track-A-only sensitivity check; non-IID RQ4 summaries use Track-A targets and require at least three.')
    a=text.index('### 7.4 RQ4:'); b=text.index('### 7.5 Run-to-Run Variability',a)
    CHANGES.append({'before':text[a:b].strip(),'after':RESULT.format(table4=table4(),table5=table5(),cap6=CAP6,cap7=CAP7).strip()})
    text=text[:a]+CHANGES[-1]['after']+'\n\n'+text[b:]
    old=next(b for b in text.split('\n\n') if b.startswith('Once federated and centralized'))
    a=old.index('Releasing the explanation vectors'); b=old.index('A natural objection',a)
    paragraph('Once federated and centralized',old[:a]+'The corrected membership-inference evaluation detects no resolved clean IID signal for the tested attack. Calibrated noise reduces explanation utility without establishing privacy mitigation; a null attack result is not a security guarantee. '+old[b:])
    paragraph('Two implications follow for anyone deploying','Two practical qualifications follow. First, the corrected RQ4 results do not identify an effective privacy budget or establish an encoding-dependent leakage ranking. The noise sweep measures explanation distortion, not certified privacy, and should not be used as a deployment protection rule. Second, failure of this particular attack to discriminate membership does not establish safe explanation release: stronger attacks, independent reference data and larger attack samples remain relevant. The RQ3 result is distinct: on Wisconsin, quantity skew stays close to IID in local-to-global distance, while moderate label skew is the most divergent functional condition; severe-skew comparisons remain confounded by collapse and client degeneracy.')
    old=next(b for b in text.split('\n\n') if b.startswith('This work extends Chen and Yoo'))
    paragraph('This work extends Chen and Yoo',old.replace('measures how much privacy budget an explanation-based attack actually requires per encoding.','evaluates empirical explanation-based membership discrimination and the utility cost of calibrated release noise, without deriving a required privacy budget.'))
    paragraph('**The direct membership-inference AUC','**The direct membership-inference AUC is an attack-specific signal-presence probe.** Fold-local preprocessing and the common background remove two identified evaluation confounds, but the attack still uses labelled explanations from the same target and a training-derived background. Only one logistic-regression attack family and 100 queried records per target are evaluated. A near-chance estimate does not rule out stronger attacks or certify privacy. Direction-aware and low-FPR diagnostics have substantial finite-sample uncertainty; historical shadow-model scores are not evidence for the corrected protocol.')
    old=next(b for b in text.split('\n\n') if b.startswith('We asked whether the explanations'))
    a=old.index('Releasing the explanation vectors'); b=old.index('Under severe label skew',a)
    paragraph('We asked whether the explanations',old[:a]+'Under the corrected release evaluation, clean direct-MIA AUCs do not establish IID membership discrimination or a leakage ranking. Calibrated Laplace noise reduces retained attribution utility, but neither effective leakage mitigation nor a formal privacy guarantee is established. These conclusions are conditional on the tested attack and finite sample sizes. '+old[b:])
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| RQ4 Privacy |'):
            new='| RQ4 Privacy | Limited: no detected clean IID signal, n=15; matched release-noise utility and Pima IID check | Limited: no detected clean IID signal, n=15; no demonstrated mitigation | Limited: no detected clean IID signal, n=15; 14 functional targets | Limited: no detected clean IID signal, n=15; isolated noisy cell remains exploratory |'
            CHANGES.append({'before':line,'after':new}); lines[i]=new
    text='\n'.join(lines)+'\n'
    p.write_text(text)
    out=ROOT/'results/rq4_release_repair_v1/promotion_paragraph_changes.json'
    out.write_text(json.dumps(CHANGES,indent=2,ensure_ascii=False)+'\n')
    caps=ROOT/'results/outputs/final/CAPTIONS.md'
    if caps.exists():
        s=caps.read_text()
        a=s.index('**Figure 6.'); b=s.index('**Figure 8.',a)
        s=s[:a]+'**Figure 6.** '+CAP6+'\n\n**Figure 7.** '+CAP7+'\n\n'+s[b:]
        s=s.replace('What does privacy cost?', 'How does release noise affect signal and utility?')
        s=s.replace('verified 0 mismatches across\nall 93 shared cells.', 'RQ4 now uses completed release_repair_v1 records and matched utility.')
        caps.write_text(s)

if __name__=='__main__': main()
