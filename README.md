# Final Project

Github Repository for Group 8's Machine Learning 2 Final Project

- [Our Proposal](Group-Proposal/ML2%20Final%20Project%20Proposal.pdf) (Now OBE)
- [Our Paper](Final-Group-Project-Report/report_team8.pdf)
- [Our Presentation](Final-Group-Presentation/presentation_team8.pdf)

## Getting the Data

Download the datasets by running the following.

```bash
# these will kick off a number of wget commands
#   simultaneously
./download_stanford_data.sh
./download_NIH_data.sh
./download_trained_models.sh

# use the following to monitor progress of downloads
#   when this lists only the grep command, the downloads
#   are complete.
ps -elf | grep wget

cd Data/
unzip CheXpert-v1.0-small.zip
unzip sample.zip
```

## Code 

All the code was ran using pycharm with the project pointing to the
root directory of the git repository, and Code repository marked as 
being a Sources Root. 

For further information about utilizing the source code please refer
to [this README file](Code/README.md).
