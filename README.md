1. Group members
   - Ahad Imran
   - Hamza Akmal Chaudary
   - Robert De La Cruz
3. Proposed project area (Perception / Behavior / Other Signals) along with proposed data types (images, RF, video, etc). 
4. Regarding (2) you are no means held to any of this and if you dont have a good idea or concept since we haven't really covered it yet...you are not behind...just do 1 and 4, while ideating on 2 for now.
5. Set up a group Git Repo and send the link to it (it does not have to be as elaborate as below, in fact it could just have README.md that has the group member names in it and nothing else).
**Group Git:** Machine Learning Setup
-- Someone init a git repo and invite collaborators (make sure everyone has keys / can push and pull)
-- Set up a README.md for everyone
-- Do something like the below as a starter kit to fill in
-- Regardless of your choice of solution you can easily start to put data in data/raw and tinker with some data/processed for EDA. You can also have a document that has all the steps you plan to take, experiments, etc. that you can start on (the HWs basically)
-- Have a discussion around commits/PRs/merges

**Project Structure and Environment**
Standardized Directory Structure: Organize the repository with clear directories to improve navigation:
- data/ (with subdirectories like raw/ and processed/)
- src/ (for source code and scripts)
- notebooks/ (for exploratory analysis)
- models/ (for trained models, and checkpoints)
- docs/ (for documentation)
- results/ (for output files like plots or metrics)

**Use Virtual Environments:** Ensure all team members use the same package versions by utilizing a virtual environment and including a requirements file (e.g., requirements.txt for Python) in the repository.
**Handle Jupyter Notebooks Carefully:** Clear notebook outputs before committing to minimize merge conflicts and repository bloat. 
**Dataloaders:** try to use https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.htmlLinks to an external site. if you want, this should expedite things down the road for you (I typically have a src/dataloader.py file) that anyone can create a new one with the same style.
