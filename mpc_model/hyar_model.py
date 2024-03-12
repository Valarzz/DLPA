import numpy as np
import torch


def three2one(d, p, par_size):
    d = d.argmax(-1)

    mask = torch.zeros([d.shape[0], par_size.max()])
    for i in range(d.shape[0]):
        offset = par_size[:d[i]].sum()
        mask[i, :par_size[d[i]]] = p[i, offset:offset+par_size[d[i]]]

    return d.unsqueeze(1), mask

def vae_train(action_rep, train_step, replay_buffer, batch_size, save_dir, vae_save_model, embed_lr, par_size):
    initial_losses = []
    for counter in range(int(train_step) + 10):
        losses = []
        # state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, not_done = replay_buffer.sample(
        #     batch_size)
        state, discrete_action, parameter_action, state_next_state, _, _ = replay_buffer.sample(batch_size)

        # discrete_action, parameter_action = three2one(discrete_action, parameter_action, par_size)
        # vae_loss, recon_loss_s, recon_loss_c, KL_loss = action_rep.unsupervised_loss(state,
        #                                             discrete_action.reshape(1, -1).squeeze().long(),
        #                                             parameter_action,
        #                                             state_next_state,
        #                                             batch_size, embed_lr)
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = action_rep.unsupervised_loss(state,
                                                    discrete_action.long(),
                                                    parameter_action,
                                                    state_next_state,
                                                    batch_size, embed_lr)
        
        losses.append(vae_loss)
        initial_losses.append(np.mean(losses))

        if counter % 1000 == 0 and counter >= 100:
            # print("load discrete embedding", action_rep.discrete_embedding())
            print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))
            # print("discrete embedding", action_rep.discrete_embedding())

        # Terminate initial phase once action representations have converged.
        if len(initial_losses) >= train_step and np.mean(initial_losses[-5:]) + 1e-5 >= np.mean(initial_losses[-10:]):
            # print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            # print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))
            # print("Converged...", len(initial_losses))
            break
        if vae_save_model:
            if counter % 1000 == 0 and counter >= 1000:
                title = "vae" + "{}".format(str(counter))
                action_rep.save(title, save_dir)
                print("vae save model")

    # state_, discrete_action_, parameter_action_, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state_, reward, not_done = replay_buffer.sample(
    #     batch_size=5000)
    
    state_, discrete_action_, parameter_action_, state_next_state_, _, _ = replay_buffer.sample(5000)
    # discrete_action_, parameter_action_ = three2one(discrete_action_, parameter_action_, par_size)

    c_rate, recon_s = action_rep.get_c_rate(state_, discrete_action_.reshape(1, -1).squeeze().long(), 
                                            parameter_action_,
                                            state_next_state_, batch_size=len(state_), range_rate=2)
    return c_rate, recon_s
