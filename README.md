# MIT-MAPLE

This repository runs the Lunar Simulator on the Alienware server computer. It includes the "MAPLE" repository within the dependencies section.

To set up the Alienware computer, enter the LAC directory:

```
cd LAC
```

Source the lac environment:

```
source lac_env/bin/activate
```

Enter the MIT-MAPLE repository (this repo - it's already installed on the computer):

```
cd MIT-MAPLE
```

Open the LunarAutonomyChallenge folder - to edit this, open this folder in a code editor like Visual Studio Code:

```
cd LunarAutonomyChallenge
```

To test/develop, create a new agent in the "agents" folder and name it 'agent0XX.py'. To test your agent, make sure to update the RunLeaderboard.sh code to export the TEAM_AGENT as your agent's new name. 
