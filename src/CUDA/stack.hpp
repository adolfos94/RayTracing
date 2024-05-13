#pragma once

#include "utilities.hpp"

namespace cuda
{
  template<class T>
  class Stack
  {
    class Node
    {
    public:
      T data;
      Node* link = nullptr;
    };

  public:

    __device__ void push(T data)
    {
      Node* newNode = new Node();
      newNode->data = data;
      newNode->link = top_;
      top_ = newNode;
    }

    __device__ bool empty() const
    {
      return top_ == nullptr;
    }

    __device__ const T top() const
    {
      if (empty())
        return T();

      return top_->data;
    }

    __device__ void pop()
    {
      if (top_ == nullptr)
        return;

      Node* temp = top_;
      top_ = top_->link;

      delete temp;

      elements--;
    }

    __device__ size_t size()
    {
      return elements;
    }

  private:

    Node* top_ = nullptr;
    size_t elements = 0;
  };
};