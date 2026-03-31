---
title: 算竞常用 C++ STL 用法
pubDatetime: 2026-03-31
description: C++ 标准模板库 (STL) 在算法竞赛中运用极其常见，本文整理了常用容器与算法的用法与注意事项。
tags: [算法, C++, STL, 竞赛]
category: 技术
draft: false
---

**C++ 标准模板库 (STL, Standard Template Library)**：包含一些常用数据结构与算法的模板的 C++ 软件库。其包含四个组件——算法 (Algorithms)、容器 (Containers)、仿函数 (Functors)、迭代器 (Iterators).

示例：

+ 算法：`sort(a.begin(), a.end())`
+ 容器：`priority_queue<int> pque`
+ 仿函数：`greater<int>()`
+ 迭代器：`vector<int>::iterator it = a.begin()`



# 1 前言
STL 作为一个封装良好，性能合格的 C++ 标准库，在算法竞赛中运用极其常见。灵活且正确使用 STL 可以节省非常多解题时间，这一点不仅是由于可以直接调用，还是因为它封装良好，可以让代码的可读性变高，解题思路更清晰，调试过程 ~~往往~~ 更顺利。

不过 STL 毕竟使用了很多复杂的结构来实现丰富的功能，它的效率往往是比不上自己手搓针对特定题目的数据结构与算法的。因此，STL 的使用相当于使用更长的运行时间换取更高的编程效率。因此，在实际比赛中要权衡 STL 的利弊，不过这一点就得靠经验了。

接下来，我会分享在算法竞赛中常用的 STL 容器和算法，对于函数和迭代器，就不着重展开讲了。



# 2 常用容器
## 2.1 内容总览
打勾的是本次将会详细讲解的，加粗的是算法竞赛中有必要学习的。

+ 顺序容器
    - [ ] **array**
    - [x] **vector**
    - [ ] **deque**
    - [ ] forward_list
    - [ ] **list**
+ 关联容器
    - [x] **set**
    - [x] **map**
    - [ ] **multiset**
    - [ ] **multimap**
+ 无序关联容器
    - [ ] **unordered_set**
    - [ ] **unordered_map**
    - [ ] **unordered_multiset**
    - [ ] **unordered_multimap**
+ 容器适配器
    - [x] **stack**
    - [x] **queue**
    - [x] **priority_queue**
    - [ ] flat_set
    - [ ] flat_map
    - [ ] flat_multiset
    - [ ] flat_multimap
+ 字符串
    - [x] **string** (basic_string<char>)
+ 对与元组
    - [x] **pair**
    - [ ] **tuple**



## 2.2 向量 vector
`#include <vector>`

连续的顺序的储存结构（和数组一样的类别），但是有长度可变的特性。

### 2.2.1 常用方法
#### 构造
`vector<类型> arr(长度, [初值])`

时间复杂度：$O(n)$

常用的一维和二维数组构造示例，高维也是一样的（就是会有点长）.

```cpp
vector<int> arr;         // 构造int数组
vector<int> arr(100);    // 构造初始长100的int数组
vector<int> arr(100, 1); // 构造初始长100的int数组，初值为1

vector<vector<int>> mat(100, vector<int> ());       // 构造初始100行，不指定列数的二维数组
vector<vector<int>> mat(100, vector<int> (666, -1)) // 构造初始100行，初始666列的二维数组，初值为-1
```

#### 尾接 & 尾删
+ `.push_back(元素)`：在 vector 尾接一个元素，数组长度 $+1$.
+ `.pop_back()`：删除 vector 尾部的一个元素，数组长度 $-1$

时间复杂度：均摊 $O(1)$

```cpp
// init: arr = []
arr.push_back(1);
// after: arr = [1]
arr.push_back(2);
// after: arr = [1, 2]
arr.pop_back();
// after: arr = [1]
arr.pop_back();
// after: arr = []
```

#### 获取长度
`.size()`：获取当前 vector 的长度，时间复杂度 $O(1)$

#### 清空 / 判空
`.clear()` 清空，$O(n)$；`.empty()` 判空，$O(1)$

#### 改变长度
`.resize(新长度, [默认值])`：修改 vector 的长度。缩短则删除多余值；扩大且指定默认值时，新元素均为默认值（旧元素不变）。时间复杂度 $O(n)$

### 2.2.2 注意事项
#### 提前指定长度
如果长度已经确定，应当直接在构造函数指定长度，而不是一个一个 `.push_back()`，可以避免重分配开销。

```cpp
// 优化前: 522ms
vector<int> a;
for (int i = 0; i < 1e8; i++)
    a.push_back(i);
// 优化后: 259ms
vector<int> a(1e8);
for (int i = 0; i < a.size(); i++)
    a[i] = i;
```

#### 当心 size_t 溢出
`.size()` 返回值类型为 `size_t`，32 位编译器下范围为 $[0,2^{32})$。

```cpp
vector<int> a(65536);
long long a = a.size() * a.size(); // 直接溢出变成0了
```

## 2.3 栈 stack
`#include <stack>`

通过二次封装双端队列 (deque) 容器，实现先进后出的栈数据结构。

| 作用 | 用法 | 示例 |
| --- | --- | --- |
| 构造 | `stack<类型> stk` | `stack<int> stk;` |
| 进栈 | `.push(元素)` | `stk.push(1);` |
| 出栈 | `.pop()` | `stk.pop();` |
| 取栈顶 | `.top()` | `int a = stk.top();` |

不可使用下标或范围 for 访问内部元素。

## 2.4 队列 queue
`#include <queue>`

通过二次封装双端队列 (deque) 容器，实现先进先出的队列数据结构。

| 作用 | 用法 | 示例 |
| --- | --- | --- |
| 构造 | `queue<类型> que` | `queue<int> que;` |
| 进队 | `.push(元素)` | `que.push(1);` |
| 出队 | `.pop()` | `que.pop();` |
| 取队首 | `.front()` | `int a = que.front();` |
| 取队尾 | `.back()` | `int a = que.back();` |

不可使用下标或范围 for 访问内部元素。

## 2.5 优先队列 priority_queue
`#include <queue>`

提供常数时间的最大元素查找，对数时间的插入与提取，底层原理是二叉堆。

#### 构造
`priority_queue<类型, 容器, 比较器> pque`

```cpp
priority_queue<int> pque1;                            // 储存int的大顶堆
priority_queue<int, vector<int>, greater<int>> pque2; // 储存int的小顶堆
```

| 作用 | 用法 |
| --- | --- |
| 进堆 | `.push(元素)` |
| 出堆 | `.pop()` |
| 取堆顶 | `.top()` |

进出堆 $O(\log n)$，取堆顶 $O(1)$。只可访问堆顶，堆中所有元素不可修改。

## 2.6 集合 set
`#include <set>`

提供对数时间的插入、删除、查找，底层原理是红黑树。

| 集合三要素 | set | multiset | unordered_set |
| --- | --- | --- | --- |
| 互异性 | ✔ | ❌（任意次） | ✔ |
| 无序性 | ❌（从小到大） | ❌（从小到大） | ✔ |

```cpp
set<int> st1;               // 储存int的集合（从小到大）
set<int, greater<int>> st2; // 储存int的集合（从大到小）
```

```cpp
// 基于范围的循环（C++11）
for (auto &ele : st)
    cout << ele << endl;
```

| 作用 | 用法 |
| --- | --- |
| 插入元素 | `.insert(元素)` |
| 删除元素 | `.erase(元素)` |
| 查找元素 | `.find(元素)` |
| 判断元素是否存在 | `.count(元素)` |

增删查时间复杂度均为 $O(\log n)$。

**注意：** set 不支持下标访问；迭代器取到的元素只读；迭代器不能相减得到下标。

## 2.7 映射 map
`#include <map>`

提供对数时间的有序键值对结构，底层原理是红黑树。

```cpp
map<int, int> mp1;               // int->int 的映射（键从小到大）
map<int, int, greater<int>> mp2; // int->int 的映射（键从大到小）
```

```cpp
// 结构化绑定 + 基于范围的循环（C++17）
for (auto &[key, val] : mp)
    cout << key << ' ' << val << endl;
```

| 作用 | 用法 |
| --- | --- |
| 增 / 改 / 查元素 | 中括号 `mp[1] = 2;` |
| 查元素（返回迭代器） | `.find(元素)` |
| 删除元素 | `.erase(元素)` |
| 判断元素是否存在 | `.count(元素)` |

增删改查时间复杂度均为 $O(\log n)$。

**注意：** 使用中括号访问不存在的键会自动插入默认值，影响键的存在性。

```cpp
map<char, int> mp;
cout << mp.count('a') << endl; // 0
mp['a'];                       // 即使什么都没做，mp['a']=0 已被插入
cout << mp.count('a') << endl; // 1
```

## 2.8 字符串 string
`#include <string>`

```cpp
string s1;           // 构造字符串，为空
string s2 = "awa!";  // 赋值 awa!
string s3(10, '6');  // 构造为 6666666666
```

| 作用 | 用法 |
| --- | --- |
| 修改、查询指定下标字符 | `[]` |
| 是否相同 | `==` |
| 字符串连接 | `+` |
| 尾接字符串 | `+=` |
| 取子串（起始下标, 子串长度） | `.substr(pos, len)` |
| 查找字符串 | `.find(str, pos)` |

数值与字符串互转（C++11）：`to_string()`、`stoi()`、`stoll()`、`stod()` 等。

**注意：** 尾接字符串必须用 `+=` 而不是 `= s + "..."` 否则性能极差；`.substr()` 第二个参数是长度不是终点下标；`.find()` 是 $O(n^2)$ 暴力实现。

## 2.9 二元组 pair
`#include <utility>`

```cpp
pair<int, char> pr = {1, 'a'}; // 列表构造 C++11
int awa = pr.first;
char bwb = pr.second;

auto &[awa, bwb] = pr; // 结构化绑定 C++17
```



# 3 迭代器简介

迭代器的作用是定义某个数据结构的遍历方式，通过迭代器便能成功遍历非线性结构（如红黑树实现的 set/map）。

```cpp
for (set<int>::iterator it = st.begin(); it != st.end(); ++it)
    cout << *it << endl;
```

+ `a.begin()`：头迭代器，指向第一个元素
+ `a.end()`：尾迭代器，指向最后一个元素**再后面一位**（不可解引用）
+ `prev(it)` / `next(it)`：返回前/后一个迭代器

**注意：** 不同容器迭代器支持的运算不同，例如 set 的迭代器不能相减求距离；在遍历时调用 `.erase()` 要格外小心迭代器失效。



# 4 常用算法
## 4.1 `swap()`
交换两个变量的值，参数为引用，无需取地址。

```cpp
int a = 0, b = 1;
swap(a, b); // a=1, b=0
```

## 4.2 `sort()`
快速排序，默认从小到大。

```cpp
vector<int> arr{1, 9, 1, 9, 8, 1, 0};
sort(arr.begin(), arr.end());                          // 从小到大
sort(arr.begin(), arr.end(), greater<int>());          // 从大到小
sort(arr.begin(), arr.end(), cmp);                     // 自定义比较器
```

自定义比较器：若 $a \star b$ 返回 `true`，若 $a = b$ **必须**返回 `false`。

## 4.3 `lower_bound()` / `upper_bound()`
在**已升序排序**的元素中二分查找，返回迭代器，找不到则返回尾迭代器。

+ `lower_bound()`：寻找 $\geq x$ 的第一个元素
+ `upper_bound()`：寻找 $> x$ 的第一个元素

```cpp
vector<int> arr{0, 1, 1, 1, 8, 9, 9};
idx = lower_bound(arr.begin(), arr.end(), 8) - arr.begin(); // 4
idx = upper_bound(arr.begin(), arr.end(), 8) - arr.begin(); // 5
```

## 4.4 `reverse()`
反转可迭代对象的元素顺序。

```cpp
reverse(arr.begin(), arr.end());
```

## 4.5 `max()` / `min()`
返回最大值/最小值，C++11 后支持列表语法。

```cpp
int mx = max({1, 2, 3, 4}); // 4
int mn = min({1, 2, 3, 4}); // 1
```

## 4.6 `unique()`
消除**相邻**重复元素，通常配合 `sort` + `erase` 使用。

```cpp
vector<int> arr{1, 2, 1, 4, 5, 4, 4};
sort(arr.begin(), arr.end());
arr.erase(unique(arr.begin(), arr.end()), arr.end());
```

## 4.7 数学函数
所有函数参数均支持 `int` / `long long` / `float` / `double` / `long double`

| 公式 | 函数 |
| --- | --- |
| $\lvert x \rvert$ | `abs(x)` |
| $e^x$ | `exp(x)` |
| $\ln x$ | `log(x)` |
| $x^y$ | `pow(x, y)` |
| $\sqrt{x}$ | `sqrt(x)` |
| $\lceil x \rceil$ | `ceil(x)` |
| $\lfloor x \rfloor$ | `floor(x)` |
| 四舍五入 | `round(x)` |

**注意浮点误差，整数操作尽量避免用浮点函数：**

+ $\lfloor a/b \rfloor$ 用 `a / b`，不要用 `floor(1.0 * a / b)`
+ $\lceil a/b \rceil$ 用 `(a + b - 1) / b`
+ $a^b$ 用快速幂，不要用 `pow(a, b)`

## 4.8 `gcd()` / `lcm()`
（C++17）返回最大公因数 / 最小公倍数。

```cpp
int x = gcd(8, 12); // 4
int y = lcm(8, 12); // 24
```

非 C++17 的 GNU 编译器可用 `__gcd()`，或自己实现欧几里得算法：

```cpp
int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
int lcm(int a, int b) { return a / gcd(a, b) * b; }
```

<div align="right">

2026年3月31日  
张骞

</div>
