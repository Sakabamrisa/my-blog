---
title: 机试输入输出总结
pubDatetime: 2026-03-31
description: 力扣和真实机试不一样的点在于力扣不用处理输入输出。这里记录一下常见的几种输入输出方法。
tags: [算法, C++, 机试]
category: 技术
draft: false
---

力扣和真实机试不一样的点在于力扣不用处理输入输出。这里记录一下常见的几种输入输出方法。

## 读有限个变量
```cpp
// scanf
int a, b, c;
scanf("%d %d %d", &a, &b, &c);

// cin
int a, b, c;
cin >> a >> b >> c;
```

## 读数量不定个变量
```cpp
// scanf
int x;
while(scanf("%d", &x) != EOF) {
    // 处理x
}

// cin
int x;
while(cin >> x) {
    // 处理x
}
```

## 读整行
```cpp
// scanf
char line[1000];
fgets(line, 1000, stdin); // 结果包含换行符

// cin
string line;
getline(cin, line); // 结果不包含换行符
```

## 混用时的坑
```cpp
// 用scanf/cin读完数字后，缓冲区残留换行符
// 紧接着读整行会读到空行！

int n;
scanf("%d", &n);   // 或者 cin >> n;
cin.ignore();      // 清除残留换行符
string line;
getline(cin, line); // 这样才能正确读到下一行
```

## 补充：大输入量加速
```cpp
// 放在main开头，用cin时防止超时
ios::sync_with_stdio(false);
cin.tie(nullptr);
// 注意：加了这句之后scanf和cin不能混用！
```

## 补充：stdin 和 cin 的本质
**stdin：**

+ 是C语言里的标准输入流，类型是 `FILE*`
+ `scanf`、`fgets` 等C函数都是从 `stdin` 读取数据
+ 本质是操作系统提供的输入缓冲区的一个指针

**cin：**

+ 是C++里的标准输入流对象，类型是 `istream`
+ `>>` 和 `getline` 都是从 `cin` 读取数据
+ 本质是对 `stdin` 的面向对象封装

**两者的关系：**

+ 默认情况下 `cin` 和 `stdin` 指向同一个缓冲区，所以混用不会丢数据
+ 但同步需要额外开销，这就是为什么加了

```cpp
ios::sync_with_stdio(false);
```

之后可以提速，但代价是**断开了同步**，此后混用 `scanf` 和 `cin` 会导致读取顺序混乱

## 输出
机试输出的本质是**把数据结构遍历一遍，按题目格式print出来**。

### 基本输出
```cpp
// printf
printf("%d\n", n);
printf("%d %d\n", a, b);

// cout
cout << n << "\n";
cout << a << " " << b << "\n";
```

### 输出数组/vector
```cpp
vector<int> ans;

for(int i = 0; i < ans.size(); i++) {
    printf("%d", ans[i]);
    if(i < ans.size()-1) printf(" "); // 末尾不加空格
}
printf("\n");
```

### 输出二维vector
```cpp
vector<vector<int>> ans;

for(int i = 0; i < ans.size(); i++) {
    for(int j = 0; j < ans[i].size(); j++) {
        printf("%d", ans[i][j]);
        if(j < ans[i].size()-1) printf(" ");
    }
    printf("\n"); // 每行换行
}
```

### 输出浮点数
```cpp
double ans;
printf("%.2f\n", ans); // 保留两位小数
```

具体格式以题目要求为准，题目说几位小数就几位小数，注意末尾有没有多余空格，这些细节可能导致WA 😊

<div align="right">

2026年3月31日  
张骞

</div>
