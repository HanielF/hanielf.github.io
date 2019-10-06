---
title: NowCoder-二叉树遍历
tags:
  - NowCoder
  - Algorithm
  - Tree
  - BinaryTree
  - Recursive
  - Medium
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-17 20:36:03
urlname: traversal-of-binary-tree
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/4b91205483694f449f94c179883c1fef?tpId=40&tqId=21342&tPage=1&rp=1&ru=/ta/kaoyan&qru=/ta/kaoyan/question-ranking)   
**补之前的**

编一个程序，读入用户输入的一串先序遍历字符串，根据此字符串建立一个二叉树（以指针方式存储）。 例如如下的先序遍历字符串： ABC##DE#G##F### 其中“#”表示的是空格，空格字符代表空树。建立起此二叉树以后，再对二叉树进行中序遍历，输出遍历结果。

### Examples:
**Input:**
输入包括1行字符串，长度不超过100。
**Output:**
可能有多组测试数据，对于每组数据，
输出将输入字符串建立二叉树后中序遍历的序列，每个字符后面都有一个空格。
每个输出结果占一行。

{% endnote %}
<!--more-->

## Solutions
- 按照先序遍历的原理，一个序列第一个肯定是根， 然后左子树。 这时候去掉了开头的第一个字符之后， 剩下的部分又相当于另一棵树
- 遍历到#的时候再返回NULL
- 关键代码是constructTree里面的递归


## C++ Codes

```C++
#include<iostream>
#include<string>
#include<cstring>
using namespace std;

struct Node{
  Node(char c):data(c), left(NULL), right(NULL){};
  char data;
  Node* left;
  Node* right;
};

Node* constructTree(string order, int &index){
  if(index<0 || index>=order.length() || order[index]=='#') return NULL;
  Node *root = new Node(order[index]);

  //这段递归很关键
  Node *left = constructTree(order, ++index);
  Node *right = constructTree(order, ++index);
  root->left = left;
  root->right = right;

  return root;
}

string travel(Node *root){
  if(root==NULL) return "";
  string res;
  string l = travel(root->left);
  string r = travel(root->right);
  res = l+root->data+r;
  return res;
}


int main(){
  string tmp;
  while(cin>>tmp){
    int pos = 0;
    Node *root = constructTree(tmp, pos);
    string res = travel(root);
    for(int i=0;i<res.length();i++){
      cout<<res[i]<<" ";
    }
    cout<<endl;
  }
  return 0;
}
```

------
