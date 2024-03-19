using Plots

function save_weights_boxplot(weights_log::Matrix{Float32}, features::Vector{String}, save_path::String)
    boxplot_data = [Float64[] for _ in 1:length(features)]
    for weight in 1:length(features)
        weight_data = [weights_log[run, weight] for run in 1:size(weights_log, 1)]
        boxplot_data[weight] = weight_data
    end

    p = boxplot(boxplot_data, labels=features, legend=false, title="Final Weight Distribution Over Runs",
                ylabel="Weight Values", palette=:Set3)
    xticks!(1:length(features), features)

    
    savefig(p, save_path)
end

# Function to create and save the diagram for results
function save_results_diagram(simulation_results::Vector{Float64}, pre_run_evaluation_result::Float64, save_path::String)
    # Create a boxplot for simulation results
    box_plot = boxplot([simulation_results], label="Simulation Runs", color=:blue, legend=:topright, width=0.5)

    # Overlay data points on the boxplot
    scatter!([ones(length(simulation_results))], simulation_results, 
             color=:orange, markersize=4, markerstrokecolor=:orange, label="Data Points")

    # Draw a horizontal line for the pre-run evaluation result
    hline!([pre_run_evaluation_result], label="Pre-run Evaluation", linestyle=:dash, color=:red)

    # Set labels and title
    xlabel!(box_plot, "Runs")
    ylabel!(box_plot, "Objective Value")
    title!(box_plot, "Objective Value Distribution Across Runs")

    # Save the figure
    savefig(box_plot, save_path)
end


function save_objective_evolution(all_rewards, save_path::String)
    num_runs = length(all_rewards)
    num_generations = maximum(length.(all_rewards))  # Get the maximum number of generations across all runs
    
    p = plot(legend=:topright)

    # Plot the curve for each simulation run
    for run in 1:num_runs
        plot!(p, 1:length(all_rewards[run]), all_rewards[run], label="Run $run", alpha=0.5)
    end
    
    title!(p, "Evolution of Objective Value Over Generations")
    xlabel!(p, "Generation")
    ylabel!(p, "Average Objective Value")
    
    savefig(p, save_path)
end


