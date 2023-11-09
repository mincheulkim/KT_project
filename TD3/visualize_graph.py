import matplotlib.pyplot as plt
import numpy as np

'''
# 1. Bar graph
x = np.arange(3)
years = ['2018', '2019', '2020']
values = [100, 400, 900]

plt.bar(x, values, color='b')
# plt.bar(x, values, color='dodgerblue')
# plt.bar(x, values, color='C2')
# plt.bar(x, values, color='#e35f62')
plt.xticks(x, years)

plt.show()


#. 2종류의 그래프 그리기
# 1. 기본 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

# 2. 데이터 준비
x = np.arange(2020, 2027)
y1 = np.array([1, 3, 7, 5, 9, 7, 14])
y2 = np.array([1, 3, 5, 7, 9, 11, 13])

# 3. 그래프 그리기
fig, ax1 = plt.subplots()

ax1.plot(x, y1, '-s', color='green', markersize=7, linewidth=5, alpha=0.7, label='Price')
ax1.set_ylim(0, 18)
ax1.set_xlabel('Year')
ax1.set_ylabel('Price ($)')
ax1.tick_params(axis='both', direction='in')

ax2 = ax1.twinx()
ax2.bar(x, y2, color='deeppink', label='Demand', alpha=0.7, width=0.7)
ax2.set_ylim(0, 18)
ax2.set_ylabel(r'Demand ($\times10^6$)')
ax2.tick_params(axis='y', direction='in')

plt.show()
'''

import matplotlib.pyplot as plt
import numpy as np


# 1. 기본 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

# 2. 데이터 준비
#x = np.arange(2020, 2027)
'''과거 자료(231024 전)'''
x = np.array([10, 8, 6, 4, 2])
y1_our = np.array([0.58, 0.54, 0.525, 0.45, 0.44])  # Ours(GP+DRL)
y1_naive = np.array([0.41, 0.325, 0.275, 0.23, 0.02])  # naive_drl
y2_our = np.array([200.74, 196, 203.51, 234.44, 237.16])
y2_naive = np.array([220.08, 219.35, 250.42, 280.04, 313.5])

'''new 자료(231024 후)'''
x = np.array([10, 8, 6, 4, 2])
y1_our = np.array([0.74, 0.70, 0.65, 0.62, 0.62])  # Ours(CIGP+DRL)  sc
y1_naive = np.array([0.56, 0.47, 0.41, 0.35, 0.30])  # naive_drl    sc
y2_our = np.array([77.67, 73.07, 64.88, 74.19, 69.6])   # t
y2_naive = np.array([69.45, 69.06, 81.27, 98.03, 110.03])   # t



# 3. 그래프 그리기
fig, ax1 = plt.subplots()
ax1.bar(x, y1_our, color='green', label='CIGP-DRL (ours)', alpha=0.7, width=0.7)  # 'deeppink'
ax1.bar(x, y1_naive, color='yellow', label='DRL [3]', alpha=1.0, width=0.7) # https://jimmy-ai.tistory.com/40
#ax1.set_ylim(0, 0.7)
ax1.set_ylim(0.2, 0.85)
ax1.set_xlabel('Sensor range (m)')
ax1.invert_xaxis()   #ax1.set_xlim(12, 0)
ax1.set_ylabel('Success rate ')
ax1.tick_params(axis='y', direction='in')



ax2 = ax1.twinx()
ax2.plot(x, y2_our, '-s', color='teal', markersize=7, linewidth=5, alpha=0.7, label='CIGP-DRL (ours)')
ax2.plot(x, y2_naive, '-s', color='orange', markersize=7, linewidth=5, alpha=0.7, label='DRL [3]')
#ax2.set_ylim(150, 350)
ax2.set_ylim(50, 150)
ax2.set_xlabel('Sensor range (m)')
ax2.set_ylabel('Average travel time')
ax2.tick_params(axis='both', direction='in')



ax2.set_zorder(ax1.get_zorder() + 10)
ax2.patch.set_visible(False)

ax2.legend(loc='upper right')
ax1.legend(loc='upper left')

plt.title('Navigation results in different sensor range')

plt.show()