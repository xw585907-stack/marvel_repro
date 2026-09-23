"""外力经关节雅可比映射时必须保持虚功；覆盖真实积分调用链。"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from envs.env import ClimbEnv


class ContactWorkTest(unittest.TestCase):
    def test_controller_feedforward_virtual_work(self):
        from stage3_mpc.control.controller import MPCController
        env = ClimbEnv(num_envs=1)
        env.phi[:] = .2
        controller = MPCController(env)
        force = np.array([[2., 3.], [-4., 1.], [1., -2.], [3., 4.]])
        controller.mpc.solve = lambda *args: force.copy()
        torque, _, _ = controller.step()
        velocity = np.array([[.3, -.2], [.4, .1], [-.1, .5], [.2, -.4]])
        foot_velocity = np.einsum('fij,fj->fi', env.get_obs()['J_w'][0], velocity)
        stance = controller.info['stance']
        np.testing.assert_allclose(np.sum(torque.reshape(4, 2)*velocity, axis=-1),
            -np.sum(force * foot_velocity, axis=-1) * stance, atol=1e-12)

    def test_external_force_virtual_work(self):
        env = ClimbEnv(num_envs=2, leg_couple_k=0.0, leg_couple_d=0.0)
        env.q += np.array([.12, -.08, -.15, .11, .04, .09, -.1, .07])
        env.phi[:] = [.2, -.3]
        force = np.array([[[2., 3.], [-4., 1.], [1., -2.], [3., 4.]]]*2)
        jac = env.get_obs()['J_w'].copy()
        # Isolate contact torque from changes in the floating base during this substep.
        env.mass[:] = 1e30
        env.inertia[:] = 1e30
        env.contact_model.step = lambda *args: force.copy()
        env._substep(np.zeros((2, 8)), np.zeros((2, 4)), env.q.copy(), np.zeros((2, 2)))
        torque = (env.qd * env.joint_inertia[:, None] / env.dt).reshape(2, 4, 2)
        virtual_velocity = np.array([[[.3, -.2], [.4, .1], [-.1, .5], [.2, -.4]]]*2)
        foot_velocity = np.einsum('nfij,nfj->nfi', jac, virtual_velocity)
        np.testing.assert_allclose(np.sum(torque * virtual_velocity, axis=-1),
                                   np.sum(force * foot_velocity, axis=-1), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
