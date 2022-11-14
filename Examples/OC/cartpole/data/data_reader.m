clear all

for i=1:7
    file = (['PDP_Neural_trial_', num2str(i-1), '.mat']);
    load(file);
    final_pos = results.solved_solution.state_traj(end,:)
end