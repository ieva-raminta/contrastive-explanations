import json
import _jsonnet

for lr in ["3e-4", "3e-5", "3e-6"]: 
    for dropout in ["0.2", "0.3", "0.4"]: 
        for hidden in ["50", "100", "200", "300"]: 
            print(lr+"_"+dropout+"_"+hidden)

            # Read the outcome.jsonnet file
            data = json.loads(_jsonnet.evaluate_file('configs/ecthr.jsonnet'))

            data["trainer"]["optimizer"]["lr"] = float(lr)
            data["model"]["dropout"] = float(dropout)
            data["model"]["feedforward"]["hidden_dims"] = int(hidden)

            # Write data into a new jsonnet file
            # named ecthr+"_"lr+"_"+dropout+"_"+hidden+".jsonnet"
            with open('configs/ecthr_'+lr+"_"+dropout+"_"+hidden+'.jsonnet', 'w') as file:
                json.dump(data, file, indent=4)