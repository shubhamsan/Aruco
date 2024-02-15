import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2 import aruco
import numpy as np
from geometry_msgs.msg import Twist
import threading
class ArucoDetectorClass(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.publisher = self.create_publisher(Image, 'aruco_image', 10)
        self.subscription = self.create_subscription(Image, 'camera/color/image_raw', self.image_callback, 10)
        self.subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()
        # self.trans_pub = self.create_publisher()

        # Camera matrices and distortion coefficients
        self.left_dist = np.array([[0., 0., 0., 0., 0.]])
        self.left_mtx = np.array([[923.429443359375,0.0,656.9810791015625],
                                  [0.,922.75537109375, 361.89886474609375],
                                  [0., 0., 1.]])

        self.right_dist = np.array([[-0.16113003, 0.00030797, 0.00048842, 0.00089638, 0.0177444]])
        self.right_mtx = np.array([[702.03209176, 0., 632.58983176],
                                   [0., 704.52835998, 371.48590454],
                                   [0., 0., 1.]])

        self.marker_length = 0.136

        


   
    def detect_aruco(self, cv_image):
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if np.all(ids != None):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.left_mtx, self.left_dist)
       
            for i in range(0, ids.size):
                aruco.drawAxis(cv_image, self.left_mtx, self.left_dist, rvec[i], tvec[i], 0.1)

            aruco.drawDetectedMarkers(cv_image, corners)
            print ("rvec: {0}, tvec: {1}".format(rvec, tvec))
            print(tvec[0][0][0])
            print(type(tvec[0][0][0]))

            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0]) + ', '

            cv2.putText(cv_image, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            return cv_image
        else:
            cv2.putText(cv_image, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            return cv_image

    def image_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        aruco_image = self.detect_aruco(cv_image)
        aruco_msg = self.cv_bridge.cv2_to_imgmsg(aruco_image, encoding='bgr8')
        #print(self.mapping_dict)
        self.publisher.publish(aruco_msg)


   



def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetectorClass()
    # Create and start the thread for your infinite loop
    rclpy.spin(aruco_detector)
    aruco_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
