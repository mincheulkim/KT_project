#!/usr/bin/env python
import sys
import cv2
import rospy
import torch
import numpy as np
import datetime

from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
from cctv_detector.msg import personinfo

class CCTVDetector(object):

    def __init__(self,
                 cctvLandmarkPoints: np.ndarray,
                 globalLandmarkPoints: np.ndarray,
                 detector_type: str = 'yolov5s'):
        self.models = {
            'detect': torch.hub.load('ultralytics/yolov5', detector_type, pretrained=True),
        }

        # detection model setting
        self.models['detect'].classes = [0]  # class filtering for yolov5. 0 means person label.

        # other settings
        self.device = torch.device('cpu')
        for m in self.models.values():
            m.eval()

        # homography calculation
        self._shape_check(cctvLandmarkPoints, 'n2')
        self._shape_check(globalLandmarkPoints, 'n2')
        self.cctv2global = self._findHomography(cctvLandmarkPoints, globalLandmarkPoints)  # [3, 3]

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> dict:
        """img -- [h, w, 3]
        """
        self._shape_check(img, 'hw3')
        outputs = {}

        # detection prediction
        outputs['detect'] = self.models['detect'](img)
        detect_xyxy = outputs['detect'].xyxy[0].cpu().numpy()
        if detect_xyxy.shape[0] == 0:
            outputs[('person_xy', 'cam')] = None
            outputs[('person_xy', 'global')] = None
            outputs['person_num'] = 0
        else:
            person_x = (detect_xyxy[:, 0] + detect_xyxy[:, 2]) * 0.5
            person_y = detect_xyxy[:, 3] - (detect_xyxy[:, 3] - detect_xyxy[:, 1]) * 0.02
            outputs[('person_xy', 'cam')] = np.stack((person_x, person_y, np.ones_like(person_y)), axis=1)
            outputs[('person_xy', 'global')] = np.einsum('kl,il->ik', self.cctv2global, outputs[('person_xy', 'cam')])
            outputs[('person_xy', 'global')] = outputs[('person_xy', 'global')] / outputs[('person_xy', 'global')][:, 2:3]  # homogeneous coords normalization
            outputs['person_num'] = outputs[('person_xy', 'cam')].shape[0]

        return outputs

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        for m in self.models.values():
            m.to(device)
        self.device = device

    def _findHomography(self, srcPoints: np.ndarray, dstPoints: np.ndarray, **kwargs) -> np.ndarray:
        """Find Homography matrix from source points and target points

        Args:
            srcPoints: source points -- [N, 2]
            dstPoints: target points -- [N, 2]

        Returns:
            Homography matrix -- [3, 3]
        """
        srcPoints = np.expand_dims(srcPoints, axis=1)  # [N, 1, 2]
        dstPoints = np.expand_dims(dstPoints, axis=1)  # [N, 1, 2]

        homography, _ = cv2.findHomography(srcPoints, dstPoints, **kwargs)

        return homography  # [3, 3]

    def _shape_check(self, tensor, target_shape: str):
        if isinstance(tensor, torch.Tensor):
            shape = tensor.size()
        elif isinstance(tensor, np.ndarray):
            shape = tensor.shape
        else:
            raise TypeError('tensor must be torch Tensor or numpy array.')

        shape = tuple(shape)
        assert len(shape) == len(target_shape), 'the dimension is not matched to each other.'

        for s, ts in zip(shape, target_shape):
            if ts.isnumeric():
                assert s == int(ts), 'the dimension is not matched to each other'

def callback(data):
    global roscv_bridge, cctv_detector, publisher
    # get image from cctv
    height, width = data.height, data.width
    img = roscv_bridge.imgmsg_to_cv2(data, "rgb8")  # mono8, mono16, bgr8, rgb8
    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)

    # detect person and publish person information
    outputs = cctv_detector(img)
    personinfo_msg = personinfo()
    personinfo_msg.date_time.data = str(datetime.datetime.now())
    personinfo_msg.num = outputs['person_num']
    if outputs['person_num'] > 0:
        personinfo_msg.idx = list(range(outputs['person_num']))
        personinfo_msg.x = outputs[('person_xy', 'global')][:, 0].tolist()
        personinfo_msg.y = outputs[('person_xy', 'global')][:, 1].tolist()
    publisher.publish(personinfo_msg)

    # publish person information to rviz
    rviz_points = Marker()
    rviz_points.header.frame_id = 'odom'
    rviz_points.id = 1
    rviz_points.type = Marker.POINTS
    rviz_points.action = Marker.ADD
    rviz_points.color = ColorRGBA(0, 1, 0, 1)
    rviz_points.scale = Vector3(0.4, 0.4, 0)

    for x, y in zip(personinfo_msg.x, personinfo_msg.y):
        rviz_points.points.append(Point(x, y, 0.5))

    rviz_publisher.publish(rviz_points)

    # print logs
    print('-------------------------------------------------')
    print(personinfo_msg.date_time.data)
    print('img_width:', data.width)
    print('img_height:', data.height)
    print('homography: \n', cctv_detector.cctv2global)
    print('person_num:\n', outputs['person_num'])
    print('person_xy (cam):\n', outputs[('person_xy', 'cam')])
    print('person_xy (global):\n', outputs[('person_xy', 'global')])

if __name__ == '__main__':
    # argument parsing
    argv = rospy.myargv(argv=sys.argv)
    cctv_name = argv[argv.index('--cam') + 1]

    cctvPoints = argv[argv.index('--camxy') + 1 : argv.index('--camxy') + 9]
    globalPoints = argv[argv.index('--globalxy') + 1 : argv.index('--globalxy') + 9]

    cctvPoints = list(map(float, cctvPoints))
    globalPoints = list(map(float, globalPoints))

    cctvPoints = np.array([
        [cctvPoints[0], cctvPoints[1]],
        [cctvPoints[2], cctvPoints[3]],
        [cctvPoints[4], cctvPoints[5]],
        [cctvPoints[6], cctvPoints[7]]
    ], dtype=np.float32)
    globalPoints = np.array([
        [globalPoints[0], globalPoints[1]],
        [globalPoints[2], globalPoints[3]],
        [globalPoints[4], globalPoints[5]],
        [globalPoints[6], globalPoints[7]]
    ], dtype=np.float32)

    print('cctvPoints:\n', cctvPoints)
    print('globalPoints:\n', globalPoints)


    # global variable declaration
    roscv_bridge = CvBridge()
    cctv_detector = CCTVDetector(cctvPoints, globalPoints)

    # node setting
    rospy.init_node('cctv_detector', anonymous=True)
    publisher = rospy.Publisher('{}/personinfo'.format(rospy.get_name()), personinfo, queue_size=10)
    rviz_publisher = rospy.Publisher('{}/personinfoviz'.format(rospy.get_name()), Marker, queue_size=1)
    rospy.Subscriber('/{}/image_raw'.format(cctv_name), Image, callback)
    rospy.spin()


