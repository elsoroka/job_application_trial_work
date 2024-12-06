import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    return mo, nn, np, plt, torch, tqdm


@app.cell
def __(nn, np):
    # parameters from paper
    N     = 5120
    dt    = 0.01 # this corresponds to 100 trades per day
    beta  = 2.0
    gamma = 0.001 # a reasonable choice
    p     = 0.5
    lm    = 0.0001 # this wasn't specified, so I left it at a reasonable value
    sigma = 0.02 # this is from page 9, section 6.1
    phi   = 0.139 # also from page 9, section 6.1
    alpha = 1.67E-4 # section 6.1

    # two more parameters, C.10
    alpha_hat = alpha * ( 1.0 - np.exp(-beta*dt) ) / ( beta*dt )
    lm_hat    = lm * ( 1.0 - np.exp(-beta*(p+1)*p*dt) ) / ( beta*(p+1)*dt )

    rng = np.random.default_rng()

    # NN setup from paper
    # NN has 2d input (f_{n-1}, J0_{n-1}) and output Qn

    class NN(nn.Module):
        def __init__(self, input_size=2, output_size=1):
            super(NN, self).__init__()
            self.layer1 = nn.Linear(input_size, 128)
            self.f1     = nn.ReLU()
            self.layer2 = nn.Linear(128, 32)
            self.f2     = nn.ReLU()
            self.layer3 = nn.Linear(32, 8)
            self.f3     = nn.ReLU()
            self.layer4 = nn.Linear(8, output_size)

            self.input_size = input_size
            self.output_size = output_size
        
        def forward(self, x):
            x = self.layer1(x)  # Apply first layer
            x = self.f1(x)    # Apply ReLU activation
            x = self.layer2(x)
            x = self.f2(x)
            x = self.layer3(x)
            x = self.f3(x)
            return self.layer4(x)

    class LinearNN(nn.Module):
        def __init__(self, input_size=2, output_size=1):
            super(LinearNN, self).__init__()
            self.layer1 = nn.Linear(input_size, output_size)

            self.input_size = input_size
            self.output_size = output_size
        
        def forward(self, x):
            return self.layer1(x)
    return (
        LinearNN,
        N,
        NN,
        alpha,
        alpha_hat,
        beta,
        dt,
        gamma,
        lm,
        lm_hat,
        p,
        phi,
        rng,
        sigma,
    )


@app.cell
def __(N, beta, dt, gamma, np, p, sigma, torch):
    def rollout(initstate, fn_seq1n, model, nsteps=N):
        # initstate is [f0, J0, Q0]
        # fn_seq1n is f1...fN
        Jn = [initstate[1]]
        Qn = [initstate[2]]
        
        # for indexing convenience, make everything indexed from n=0 to n=N
        Jn0 = Jn[0] # assumption: initialize J_n^{0,\theta} this way
        
        for t in range(1,nsteps+1):
            input = torch.FloatTensor([fn_seq1n[t-1,0], Jn0])
            Qn.append(model(input)) # C.14, this sets Qn
            Jn0 = np.exp(-beta*dt)*Jn0 - (1.0-np.exp(-beta*dt))*Qn[-1] # C.15
            Jn.append((Qn[-1] + Jn0)) # C.16

        # new states
        return torch.hstack([torch.FloatTensor(fn_seq1n), torch.vstack(Jn), torch.vstack(Qn)])

    # the reward of one policy rollout
    def reward(states, alpha_hat, lm_hat, ):
        fns = torch.squeeze(states[:-1,0]) # f_0...f_{N-1}
        Jns = torch.squeeze(states[1:,1]) # J1...JN
        Qns = torch.squeeze(states[1:,2]) # Q1...QN
        # C.17
        return torch.sum( alpha_hat*Qns*fns - gamma*sigma**2/2.0 * torch.square(Qns) - lm_hat*beta*torch.abs(Jns)**(p+1) )

    return reward, rollout


@app.cell
def __(mo):
    mo.md(
        """
        ## Initialization
        This defines the initializes given in Appendix C.2
        """
    )
    return


@app.cell
def __(dt, np, phi, rng):
    def random_init(cov, batch_size=1024):
        # returns batch_size x n matrix of random samples
        # for convenience define this as [fn, Jn, Qn] because this is easier to index into
        return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=batch_size)

    # this simulates the batch of f_n's with given length N
    def random_signal_batch(init_fn, batch_size=1024, N=5120):
        # init_fn is a batch_size vector of initial f1's
        signals = np.zeros((N+1,batch_size))
        # for efficiency, we advance the whole batch_size of f_{n-1} to f_n
        # we could make this more efficient if needed
        signals[0,:] = init_fn
        for i in range(1, N+1):
            signals[i,:] = np.exp(-phi*dt)*signals[i-1,:] + np.sqrt(1.0 - np.exp(-2.0*phi*dt)) * rng.normal(size=batch_size)
        return signals

    def update_cov(signals, prev_cov):
        # signals is a T x m matrix where T = N*batch_size and m = number of states (3)
        sample = np.cov(signals, rowvar=False) # estimate cov of signals
        return prev_cov*0.999 + 0.001*sample
    return random_init, random_signal_batch, update_cov


@app.cell
def __(mo):
    mo.md(
        """
        ## Quick check on setup
        Mostly we want to see that the $f_n$'s look reasonable
        """
    )
    return


@app.cell
def __(np, plt, random_init, random_signal_batch):
    initstates_test1 = random_init(np.identity(3), batch_size=100)
    fns_test1 = random_signal_batch(initstates_test1[:,0], batch_size=100)
    for i in range(10):
        p1 = plt.plot(fns_test1[:,i])
        plt.title("10 sampled signals $f_n$")
        plt.xlabel("Step n")
    plt.savefig("f1_sample.pdf")
    plt.show()
    return fns_test1, i, initstates_test1, p1


@app.cell
def __(mo):
    mo.md("""## Neural network training""")
    return


@app.cell
def __(NN):
    model = NN(input_size=2, output_size=1)
    N_EPOCHS = 50
    BATCH_SIZE = 16
    return BATCH_SIZE, N_EPOCHS, model


@app.cell
def __(model, torch):
    # learning rate is 1e-4 or 1e-2 for the linear network for 30 epochs, then reduced by 1/4 every 5 epochs after
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Learning rate scheduler defined in paper
    scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=30)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)
    return optimizer, scheduler, scheduler1, scheduler2


@app.cell
def __(
    BATCH_SIZE,
    N,
    N_EPOCHS,
    alpha_hat,
    lm_hat,
    np,
    random_init,
    random_signal_batch,
    reward,
    rollout,
    torch,
    tqdm,
    update_cov,
):
    def train_nn(model, optimizer, scheduler, BATCH_SIZE=BATCH_SIZE, N_EPOCHS=N_EPOCHS):
        cov = np.identity(3) # there isn't an initialization given
        validation_reward = np.zeros(N_EPOCHS)
        
        for e in range(N_EPOCHS):
            # set up batch
            initstates = random_init(cov, batch_size=BATCH_SIZE)
            fns = random_signal_batch(initstates[:,0], batch_size=BATCH_SIZE)
            initstates = torch.FloatTensor(initstates)
            fns = torch.FloatTensor(fns)
        
            optimizer.zero_grad()
            
            for t in tqdm(range(BATCH_SIZE)):
                sequence = rollout(initstates[t,:], fns[:,t:t+1], model, nsteps=N)
                loss = -reward(sequence, alpha_hat, lm_hat)
                loss.backward()
                
                cov = update_cov(sequence.detach(), cov)
            
            # optimize
            optimizer.step()
            scheduler.step()
        
            with torch.no_grad():
                # now we have finished a batch
                validation_initstates = random_init(cov, batch_size=BATCH_SIZE)
                validation_fns = random_signal_batch(validation_initstates[:,0], batch_size=BATCH_SIZE)
                validation_initstates = torch.FloatTensor(validation_initstates)
                validation_fns = torch.FloatTensor(validation_fns)
                
                sequence = rollout(validation_initstates[t,:], validation_fns[:,t:t+1], model, nsteps=N)
                validation_reward[e] = reward(sequence, alpha_hat, lm_hat)
                print(f"Epoch {e}: reward {validation_reward[e]}")
        
        return validation_reward, cov
    return (train_nn,)


@app.cell
def __(model, optimizer, plt, scheduler, train_nn):
    validation_reward, cov = train_nn(model, optimizer, scheduler)

    plt.plot(validation_reward, label="One layer NN")
    plt.ylim([-2.0, 12.0])
    plt.title("Linear NN reward")
    plt.xlabel("Epoch")
    plt.savefig("linear_training_reward.pdf")
    plt.show()
    return cov, validation_reward


@app.cell
def __(
    N,
    cov,
    model,
    plt,
    random_init,
    random_signal_batch,
    rollout,
    torch,
):
    seq = None
    with torch.no_grad():
        initstates_test = random_init(cov, batch_size=100)
        fns_test = random_signal_batch(initstates_test[:,0], batch_size=100)

        seq = rollout(torch.FloatTensor(initstates_test[0,:]), fns_test[:,0:1], model, nsteps=N)

    # visualizing
    plt.plot(seq[:,0], label="{fn} (sample)")
    plt.plot(seq[:,1], label="{J_n} (sample)")
    plt.plot(seq[:,2], label="{Qn} (sample)")
    plt.title("One policy rollout")
    plt.xlabel("Step n")
    plt.legend()
    plt.show()
    return fns_test, initstates_test, seq


@app.cell
def __(LinearNN, plt, torch, train_nn):
    model2 = LinearNN(input_size=2, output_size=1)
    # learning rate is 1e-4 or 1e-2 for the linear network for 30 epochs, then reduced by 1/4 every 5 epochs after
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-2)

    # Learning rate scheduler defined in paper
    scheduler12 = torch.optim.lr_scheduler.ConstantLR(optimizer2, factor=1.0, total_iters=30)
    scheduler22 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.75)
    linear_scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler12, scheduler22], optimizer=optimizer2)

    validation_reward2, cov2 = train_nn(model2, optimizer2, linear_scheduler)

    plt.plot(validation_reward2, label="One layer NN")
    plt.ylim([-2.0, 12.0])
    plt.title("Linear NN reward")
    plt.xlabel("Epoch")
    plt.savefig("linear_training_reward.pdf")
    plt.show()
    return (
        cov2,
        linear_scheduler,
        model2,
        optimizer2,
        scheduler12,
        scheduler22,
        validation_reward2,
    )


@app.cell
def __(N, fns_test, initstates_test, model2, plt, rollout, seq, torch):
    seq2 = None
    with torch.no_grad():

        seq2 = rollout(torch.FloatTensor(initstates_test[0,:]), fns_test[:,0:1], model2, nsteps=N)

    # visualizing
    plt.plot(seq[:,0], label="{fn} (sample)")
    plt.plot(seq[:,1], label="{J_n} (sample)")
    plt.plot(seq[:,2], label="{Qn} (sample)")
    plt.title("One single layer policy rollout")
    plt.xlabel("Step n")
    plt.legend()
    plt.show()
    return (seq2,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Results

        * The neural network successfully trains for 50 epochs, although it is noisy. It looks similar to the linear network loss from the paper, which starts higher than the other networks and increases slowly with some noise. However there is a scaling difference: the loss from the paper is scaled up from mine.
        * I am not sure what the policy rollout should look like; however it looks like the trading signal $Q_n$ correlates with $J_n^\theta$ and $f_n$. I also notice that $Q_n$ changes when $f_n$ is negative. This plot wasn't requested; I just wanted to see it.
        * The linear network $NN_\theta = L_{2,1}$ doesn't work as well, likely because it doesn't have enough parameters to learn a meaningful predictive function.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### Further discussion: this code is slow.
        I wanted to know why it was slow, so I profiled it.
        Inspecting the profile results shows that about 50% of the time is spent in the policy rollout, and 50% on the gradient backward() computation.

        * There is probably a more efficient way to implement the rollout.
        * The main efficiency gain we could make is by **parallelizing the batch**, which is what I would do with more implementation time.
        """
    )
    return


@app.cell
def __():
    """
    def test(cov):
        initstates_test = random_init(cov, batch_size=100)
        fns_test = random_signal_batch(initstates_test[:,0], batch_size=100)

        seq = rollout(torch.FloatTensor(initstates_test[0,:]), torch.FloatTensor(fns_test[:,0:1]), model, nsteps=N)
        l = -reward(seq, alpha_hat, lm_hat)
        l.backward()
        
    import cProfile
    cProfile.run('test(cov)', 'restats')
    """
    return


@app.cell
def __():
    """
    import pstats
    from pstats import SortKey
    ps = pstats.Stats('restats')
    ps.sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()
    """
    return


if __name__ == "__main__":
    app.run()
