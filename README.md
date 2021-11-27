# ev-table-dlrm
Modifying embedding layers for dlrm

## Setup instance
Setup Instance for dlrm

### Preparation [Login as "cc"]

    Use cc user!!
    ssh cc@129.114.108.229 (take note on your floating ip address)

    Setup disk 
        # check if there is already mounted disk
        df -H
            # /dev/sda1       251G  2.8G  248G   2% /
            # should be enough

    # Setup user kahfi 
    sudo adduser kahfi
    sudo usermod -aG wheel kahfi
    sudo su 
    cp -r /home/cc/.ssh /home/kahfi
    chmod 700  /home/kahfi/.ssh
    chmod 644  /home/kahfi/.ssh/authorized_keys
    chown kahfi  /home/kahfi/.ssh
    chown kahfi  /home/kahfi/.ssh/authorized_keys
    echo "kahfi ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/90-cloud-init-users
    exit
    exit

### Setup zsh [Login on "kahfi"], don't have to, but recommended

    ssh 129.114.108.229 

        sudo su
        apt-get install zsh -y
        chsh -s /bin/zsh root
        chsh -s /bin/zsh kahfi
        exit
        which zsh
        echo $SHELL

        sudo apt-get install wget git vim zsh -y

        printf 'Y' | sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

        /bin/cp /.oh-my-zsh/templates/zshrc.zsh-template /.zshrc
        sudo sed -i 's|home/kahfi:/bin/bash|home/kahfi:/bin/zsh|g' /etc/passwd
        sudo sed -i 's|ZSH_THEME="robbyrussell"|ZSH_THEME="risto"|g' ~/.zshrc
        zsh
        # sudo source ~/.zshrc

### Environment setup
    
        # Install anaconda3 [Run Once]
        sudo mkdir -p /mnt/extra
        sudo chown kahfi -R /mnt
        cd /mnt/extra
        wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh  --no-check-certificate
        bash Anaconda3-5.3.0-Linux-x86_64.sh -b -p $HOME/anaconda3
        # zsh activate conda 
        echo "export PATH=\"~/anaconda3/bin:\$PATH\"" >> ~/.zshrc
        cd ~/anaconda3/bin && ./conda update conda -y
        ./conda init zsh 
        exit

        ssh 129.114.108.229 


        condaEnvName=“dlrm”
        conda create --name dlrm python=3.7 -y
        conda activate dlrm
        echo "conda activate dlrm" >> ~/.zshrc

        which -a pip | grep dlrm
            # get anaconda3's pip for the $condaEnvName
            # Anything installed by this $condaPip will only be available to $condaEnvName
        export condaPip=/home/kahfi/anaconda3/envs/dlrm/bin/pip
        
        $condaPip --version
        # make sure it output the version (i.e: pip 21.0.1 from /home/daniar/anaconda3/envs/dlrm/lib/python3.7/site-packages/pip (python 3.7))

        # Now, install the rest of the dependencies
	      $condaPip install future
        $condaPip install numpy
        $condaPip install pandas
        $condaPip install scikit-learn
        $condaPip install onnx
	      $condaPip install torchviz
	      $condaPip install mpi
	      $condaPip install torch
	      $condaPip install tqdm
	      $condaPip install pydot
        
	git clone https://github.com/mlperf/logging.git mlperf-logging
	$condaPip install -e mlperf-logging
	$condaPip install tensorboard

        # conda deactivate --> To quit the current conda environment
        # conda env list   --> To list ALL available conda environment in your machine
        # conda remove --name <env_NAME> --all
    # Install Jupyter Notebook at server
        # Jupyter Notebook is a web-based user interface for easier operation on the server 
        # e.g. edit file, delete file, move file, etc.

        # Install Jupyter and ipykernel on Anaconda environment
        $condaPip install jupyter
        $condaPip install ipykernel

        # Add conda env to jupyter
        condaEnvName="dlrm"
        python -m ipykernel install --user --name=$condaEnvName
        
        # Install jupyterthemes
        $condaPip install jupyterthemes

        # Set theme
        jt -l
        jt -t chesterish

        # Launch Jupyter Notebook
        cd $GITHUB 
        nohup jupyter notebook &
        sleep 3 
        tail nohup.out

            # Output sample (Pleasae save the token) :
            http://localhost:8889/?token=386a3846dee6fb25816cd75a4a7ed55bacc2247c49d2803a

    # Start editing the Jupyter Notebook [Run at LOCAL]
        
        # Port forwarding
        # Forward the web server port to your local machine 
        # Open a new terminal then run the script below. 
        # Also, NEVER close this terminal as long as you use Jupyter Notebook 
        
        ssh -L 9999:localhost:8888 <uusername>@<Server's Public IP>
        
        # Example : ssh -L 9999:localhost:8888  cc@129.114.109.84
        # To access the notebook, open this URL and add the token from the previous step 
        
        http://localhost:9999/
	
## Setting up the dataset

	#Open terminal from ev-table-dlrm directory
	mkdir input
	cd input
	wget http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
	tar -xzvf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
	
## Running the DLRM model
### Create additional folders for the DLRM

	mkdir weights_and_biases
	cd weights_and_biases
	mkdir epoch-0
	cd epoch-0
	mkdir ev-table
	cd ../..
	
### Now save the model and the weights

	nohup ./bench/dlrm_s_criteo_kaggle.sh "--save-model=model.pth --test-freq=153485 --nepochs=1 --mlperf-logging" &
	
### Truncate the weights 

	./reduce_precision.py -file weights_and_biases/epoch-0/ev-table-1.csv -read_as fp32 -new_precision 4 
	#new_precision can be {4, 8, 16, 32}
	
### Load the new weights

	nohup ./bench/dlrm_s_criteo_kaggle.sh "--load-model=model.pth --ev-path=weights_and_biases/epoch-0/ev-table-4 --inference-only --mlperf-logging" & 
	
From the nohup.out file you should get the ROC value
