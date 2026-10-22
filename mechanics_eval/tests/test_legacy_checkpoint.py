"""Legacy checkpoint dispatch and precision must preserve the original model interface."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mechanics_eval.rollouts import ModuleAdapter


class LegacyCheckpointTests(unittest.TestCase):
    def test_schema_dispatch_uses_contents_not_filename(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            legacy=root/'arbitrary-name.npz';np.savez(legacy,meta='{}',arr_0=np.zeros(1))
            current=root/'iteration1.npz';np.savez(current,meta='{}',**{'enc.0':np.zeros(1)})
            module=SimpleNamespace(load_model=lambda path:object(),start=lambda *a:None,step=lambda *a:None)
            for path,expected in [(legacy,'train_tiny'),(current,'train')]:
                with patch('mechanics_eval.rollouts.importlib.import_module',return_value=module) as load:
                    ModuleAdapter('train',path)
                    load.assert_called_once_with(expected)
            with patch('mechanics_eval.rollouts.importlib.import_module',return_value=module) as load:
                ModuleAdapter('custom_adapter',legacy)
                load.assert_called_once_with('custom_adapter')

    def test_explicit_legacy_module_does_not_require_current_checkpoint_config(self):
        module=SimpleNamespace(load_model=lambda path:object(),start=lambda *a:None,step=lambda *a:None)
        with patch('mechanics_eval.rollouts.importlib.import_module',return_value=module) as load:
            ModuleAdapter('train_tiny',Path('not-read-by-custom-loader.npz'))
            load.assert_called_once_with('train_tiny')

    def test_fp32_changes_only_loaded_legacy_layer_compute_dtypes(self):
        import jax.numpy as jnp
        from flax import nnx
        class Model(nnx.Module):
            def __init__(self):
                self.linear=nnx.Linear(2,2,dtype=jnp.bfloat16,rngs=nnx.Rngs(0))
                self.conv=nnx.Conv(2,2,(1,1),dtype=jnp.bfloat16,rngs=nnx.Rngs(0))
                self.up=nnx.ConvTranspose(2,2,(1,1),dtype=jnp.bfloat16,rngs=nnx.Rngs(0))
        created=[]
        def loader(path):
            model=Model();created.append(model);return model
        module=SimpleNamespace(load_model=loader,start=lambda *a:None,step=lambda *a:None)
        with patch('mechanics_eval.rollouts.importlib.import_module',return_value=module):
            fp32=ModuleAdapter('train_tiny',Path('fixture'),True)
            native=ModuleAdapter('train_tiny',Path('fixture'),False)
        for name in ('linear','conv','up'):
            self.assertEqual(getattr(fp32.model,name).dtype,jnp.float32)
            self.assertEqual(getattr(native.model,name).dtype,jnp.bfloat16)
        from jax.tree import leaves
        for a,b in zip(leaves(nnx.state(fp32.model)),leaves(nnx.state(native.model))):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
