"""Read-only parity against 100 trusted original checkpoints plus head math checks."""
import json
import numpy as np
import pandas as pd
import torch
from .data import Bundle
from ._numeric_encoder import Encoder
from .models import make_model, feature_digest
from .training import FittedModel
from .provenance import original_fold
from .time_grid import TimeGrid
from .losses.discrete import logistic_hazard_nll
from .prediction import discrete_predictions
from .storage import write_json
from .runtime import stamp


def verify_original(study):
    common=study.config["common"];rows=[]
    if not torch.cuda.is_available():raise RuntimeError("Same CUDA device required for reference parity")
    for fold in range(1,6):
        bundle=Bundle(study.bundle_path(fold));base=original_fold(study,fold)
        expected=pd.read_csv(base/"predictions_all20.csv",float_precision="round_trip")
        for j in range(1,21):
            path=base/f"imputation_{j:02d}/cacox.pt"
            # Trusted local pipeline checkpoint; hash verified before this call.
            obj=torch.load(path,map_location="cpu",weights_only=False)
            enc=Encoder(obj["encoder"]["variables"],common["scale"]);enc.__dict__.update(obj["encoder"])
            net=make_model(len(enc.columns),study.config["control"],common,obj["seed"])
            init_digest=feature_digest(net)
            other=make_model(len(enc.columns),study.config["candidate"],common,obj["seed"])
            if feature_digest(other)!=init_digest:raise AssertionError("Head changed original backbone initialization")
            del other
            net.load_state_dict(obj["state_dict"])
            model=FittedModel(net,enc,study.config["control"],common,None,obj["baseline"])
            _,assessment=bundle.pair(j);score,survival=model.predict(assessment)
            target=expected[expected.imputation.eq(j)].set_index("patient_id").loc[assessment.patient_id]
            error=float(np.max(np.abs(score-target.score.to_numpy(float))))
            np.testing.assert_allclose(score,target.score.to_numpy(float),rtol=0,atol=1e-7)
            for k,t in enumerate(common["times"]):
                np.testing.assert_allclose(survival[:,k],target[f"survival_{t}d"].to_numpy(float),rtol=0,atol=1e-7)
            rows.append(dict(fold=fold,imputation=j,max_score_error=error,common_feature_initialization_equal=True))
            del model,net
    # Analytical censoring likelihood and gradients, not a copied prediction test.
    grid=TimeGrid(np.array([0.,1.,2.,3.]))
    labels=grid.labels([1,2,1,1.5,.5,4,3],[1,1,0,0,0,1,1])
    logits=torch.zeros(labels[0].shape,requires_grad=True)
    value=logistic_hazard_nll(logits,labels);value.backward()
    torch.testing.assert_close(logits.grad,(.5-labels[1])*labels[0]/7)
    assert abs(float(value.detach())-float(labels[0].sum())*np.log(2)/7)<1e-6
    grid=TimeGrid(np.array([0.,200.,700.,1500.]));rate=.001
    logits=np.log(np.expm1(rate*np.diff(grid.cuts)))[None,:]
    score,survival=discrete_predictions(logits,grid)
    np.testing.assert_allclose(survival[0],np.exp(-rate*np.array([365,730,1095])),atol=1e-12)
    np.testing.assert_allclose(score[0],np.expm1(-rate*1095)/rate,atol=1e-10)
    pd.DataFrame(rows).to_csv(study.out/"validation/original_checkpoint_equivalence.csv",index=False)
    report=dict(status="passed",at=stamp(),checkpoints=len(rows),max_stored_gpu_error=max(q["max_score_error"] for q in rows),
                original_network_exact_copy=True,feature_initialization_equal_all100=True,discrete_likelihood_gradient_and_survival="passed")
    write_json(study.out/"validation/ORIGINAL_EQUIVALENCE.json",report)
    return report
