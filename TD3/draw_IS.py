import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline

# 주어진 데이터
data = np.array([
    [-1.49753809, 0.23718673],
    [-1.56517318, -0.24789908],
    [-1.4678731, -0.7479187],
    [-1.18330531, -1.18330531],
    [-0.77211658, -1.51536410],
    [-0.27829975, -1.75711548],
    [0.29606560, -1.86928462],
    [0.90902588, -1.78406375],
    [1.44924711, -1.44924711],
    [1.78406375, -0.90902588],
    [1.86928462, -0.29606560],
    [1.75711548, 0.27829975],
    [1.51536410, 0.77211658],
    [1.18330531, 1.18330531],
    [0.74791870, 1.46787310],
    [0.24789908, 1.56517318],
    [-0.23718673, 1.49753809],
    [-0.66594140, 1.30698359],
    [-1.02477246, 1.02477246],
    [-1.30698359, 0.66594140]])


# Convex Hull 계산
hull = ConvexHull(data)

# Convex Hull 경계 점 좌표
hull_points = np.append(data[hull.vertices], [data[hull.vertices[0]]], axis=0)

# 부드럽게 표현하기 위한 Cubic Spline 사용
t = np.arange(hull_points.shape[0])
t_new = np.linspace(0, t.max(), 300)
cs_x = CubicSpline(t, hull_points[:, 0])
cs_y = CubicSpline(t, hull_points[:, 1])

# 부드러운 곡선을 만들기 위한 좌표 계산
smooth_x = cs_x(t_new)
smooth_y = cs_y(t_new)


# 그래프 표시
plt.figure(figsize=(5.5, 5.5))
#plt.plot(smooth_x, smooth_y, (255,255,250), lw=2)
#plt.plot(sommth_x_2, sommth_y_2, 'b-', lw=2)
for i in range(5):
    plt.plot(smooth_x*(1-0.1*i), smooth_y*(1-0.1*i), 'b-', lw=1)    
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title('Individual Space (IS)')
plt.grid(True)
plt.show()

