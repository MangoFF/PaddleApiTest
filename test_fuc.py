import numpy as np
def test_equl(out_eager_np,out_torch,rtol=1e-7, atol=0,err_str='',dtype = None):
    max_atol_idx = np.argmax(np.abs(out_eager_np-out_torch))
    max_rtol_idx = np.argmax(np.abs((out_eager_np-out_torch)/out_eager_np))
# compare eager res with torch
    np.testing.assert_allclose(
        out_eager_np,
        out_torch,
        atol,
        rtol,
        err_msg=
        err_str   
        % (
        dtype,
        max_atol_idx, 
        out_eager_np.flatten()[max_atol_idx].item(), 
        out_torch.flatten()[max_atol_idx].item(),
        max_rtol_idx, 
        out_eager_np.flatten()[max_rtol_idx].item(), 
        out_torch.flatten()[max_rtol_idx].item()),
    )
def test_equls(out_grads_eager_np,out_grads_torch,rtol=1e-7, atol=0,err_str='',dtype = None):
    for idx in range(len(out_grads_eager_np)):
        max_atol_idx = np.argmax(np.abs(out_grads_eager_np[idx]-out_grads_torch[idx]))
        max_rtol_idx = np.argmax(np.abs((out_grads_eager_np[idx]-out_grads_torch[idx])/out_grads_eager_np[idx]))
        np.testing.assert_allclose(
            out_grads_eager_np[idx],
            out_grads_torch[idx],
            atol,
            rtol,
            err_msg=
            err_str
            % (
            dtype, 
            max_atol_idx, 
            out_grads_eager_np[idx].flatten()[max_atol_idx].item(),
            out_grads_torch[idx].flatten()[max_atol_idx].item(),
            max_rtol_idx,
            out_grads_eager_np[idx].flatten()[max_rtol_idx].item(), 
            out_grads_torch[idx].flatten()[max_rtol_idx].item()),
        )