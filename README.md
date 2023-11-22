# KT_project
KT oneteam project about Indoor robot navigation

1. 터미널 두개 열기
2. 각 터미널에서 모두 다음 명령어 실행
   export ROS_HOSTNAME=localhost
   export ROS_PORT_SIM=11311
   export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
   export GAZEBO_MODEL_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
   source ~/.bashrc
   cd ~/DRL-robot-navigation/catkin_ws
   source devel_isolated/setup.bash
4. 1번 터미널에서 다음 명령어 순차 실행
   cd ~/DRL-robot-navigation/TD3
   python train_velodyne_td3_sac.py
5. 2번 터미널에서 다음 실행
   roslaunch pedsim_simulator security.launch
6. 프로세스 종료
   killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3 rviz
