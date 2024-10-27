#pragma once

#include <initializer_list>
#include <algorithm>
#include <deque>
#include <iostream>

class Block {
public:
    static const size_t kSize = 512 / sizeof(int);

    Block() {
        std::fill(data_, data_ + kSize, 0);
        last_ = nullptr;
        next_ = nullptr;
    }

    ~Block() {
    }

    int& operator[](size_t index) {
        return data_[index];
    }

    friend class Deque;

private:
    int data_[kSize];
    Block* last_;
    Block* next_;
};

class Deque {
public:
    Deque() {
        size_ = 0;
        capacity_ = Block::kSize;
        first_block_ = new Block();
        first_block_->last_ = first_block_;
        first_block_->next_ = first_block_;
        last_block_ = first_block_;
        first_index_ = 0;
        last_index_ = 0;
    }

    Deque(const Deque& rhs) : Deque() {
        for (size_t i = 0; i < rhs.size_; ++i) {
            PushBack(rhs[i]);
        }
    }

    Deque(Deque&& rhs) {
        size_ = 0;
        capacity_ = Block::kSize;
        first_block_ = new Block();
        first_block_->last_ = first_block_;
        first_block_->next_ = first_block_;
        last_block_ = first_block_;
        first_index_ = -1;
        last_index_ = -1;
        for (size_t i = 0; i < rhs.size_; ++i) {
            PushBack(rhs[i]);
        }
        rhs.Clear();
    }

    explicit Deque(size_t size) : Deque() {
        for (size_t i = 0; i < size; ++i) {
            PushBack(0);
        }
    }

    Deque(std::initializer_list<int> list) : Deque() {
        for (int value : list) {
            PushBack(value);
        }
    }

    ~Deque() {
        Clear();
        Block* cur_block = first_block_;
        for (size_t i = 0; i < capacity_ / Block::kSize; ++i) {
            Block* next = cur_block->next_;
            delete cur_block;
            cur_block = next;
        }
    }

    Deque& operator=(Deque rhs) {
        bool equal = true;
        if (rhs.size_ != size_) {
            equal = false;
        } else {
            for (size_t i = 0; i < size_; ++i) {
                if (Get(i) != rhs[i]) {
                    equal = false;
                    break;
                }
            }
        }
        if (equal) {
            return *this;
        }
        Clear();
        for (size_t i = 0; i < rhs.size_; ++i) {
            PushBack(rhs[i]);
        }
        return *this;
    }

    void Swap(Deque& rhs) {
        std::swap(size_, rhs.size_);
        std::swap(capacity_, rhs.capacity_);
        std::swap(first_block_, rhs.first_block_);
        std::swap(last_block_, rhs.last_block_);
        std::swap(first_index_, rhs.first_index_);
        std::swap(last_index_, rhs.last_index_);
    }

    void PushBack(int value) {
        if (size_ == capacity_) {
            Reallocate();
        }
        if (last_index_ == Block::kSize - 1 && last_block_->next_ == first_block_) {
            Reallocate();
        }
        ++size_;
        if (size_ == 1) {
            first_index_ = 0;
            last_index_ = 0;
            first_block_->data_[first_index_] = value;
            return;
        }
        MoveRight(last_block_, last_index_);
        last_block_->data_[last_index_] = value;
    }

    void PopBack() {
        --size_;
        if (size_ == 0) {
            first_index_ = 0;
            last_index_ = 0;
            return;
        }
        MoveLeft(last_block_, last_index_);
    }

    void PushFront(int value) {
        if (size_ == capacity_) {
            Reallocate();
        }
        if (first_index_ == 0 && first_block_->last_ == last_block_) {
            Reallocate();
        }
        ++size_;
        if (size_ == 1) {
            first_index_ = 0;
            last_index_ = 0;
            first_block_->data_[first_index_] = value;
            return;
        }
        MoveLeft(first_block_, first_index_);
        first_block_->data_[first_index_] = value;
    }

    void PopFront() {
        --size_;
        if (size_ == 0) {
            first_index_ = 0;
            last_index_ = 0;
            return;
        }
        MoveRight(first_block_, first_index_);
    }

    int Get(size_t ind) {
        size_t el_in_first_block = Block::kSize - first_index_;
        if (ind < el_in_first_block) {
            return first_block_->data_[ind + first_index_];
        }
        ind -= el_in_first_block;
        Block* cur_block = first_block_->next_;
        for (size_t i = 0; i < ind / Block::kSize; ++i) {
            cur_block = cur_block->next_;
        }
        return cur_block->data_[ind % Block::kSize];
    }

    int& operator[](size_t ind) {
        size_t el_in_first_block = Block::kSize - first_index_;
        if (ind < el_in_first_block) {
            return first_block_->data_[ind + first_index_];
        }
        ind -= el_in_first_block;
        Block* cur_block = first_block_->next_;
        for (size_t i = 0; i < ind / Block::kSize; ++i) {
            cur_block = cur_block->next_;
        }
        return cur_block->data_[ind % Block::kSize];
    }

    int operator[](size_t ind) const {
        size_t el_in_first_block = Block::kSize - first_index_;
        if (ind < el_in_first_block) {
            return first_block_->data_[ind + first_index_];
        }
        ind -= el_in_first_block;
        Block* cur_block = first_block_->next_;
        for (size_t i = 0; i < ind / Block::kSize; ++i) {
            cur_block = cur_block->next_;
        }
        return cur_block->data_[ind % Block::kSize];
    }

    size_t Size() const {
        return size_;
    }

    void Clear() {
        while (size_) {
            PopBack();
        }
    }

    void MoveLeft(Block*& cur_block, size_t& cur_index) {
        if (cur_index == 0) {
            cur_block = cur_block->last_;
            cur_index = Block::kSize - 1;
        } else {
            --cur_index;
        }
    }

    void MoveRight(Block*& cur_block, size_t& cur_index) {
        if (cur_index == Block::kSize - 1) {
            cur_block = cur_block->next_;
            cur_index = 0;
        } else {
            ++cur_index;
        }
    }

    void Reallocate() {
        capacity_ *= 2;
        if (first_block_ == last_block_) {
            Block* new_block = new Block();
            new_block->last_ = first_block_;
            new_block->next_ = first_block_;
            first_block_->last_ = new_block;
            first_block_->next_ = new_block;
            return;
        }
        Block* last_created = last_block_;
        for (size_t i = 0; i < capacity_ / (2 * Block::kSize); ++i) {
            Block* new_block = new Block();
            last_created->next_ = new_block;
            new_block->last_ = last_created;
            last_created = new_block;
        }
        last_created->next_ = first_block_;
        first_block_->last_ = last_created;
    }

private:
    size_t size_;
    size_t capacity_;
    Block* first_block_;
    Block* last_block_;
    size_t first_index_;
    size_t last_index_;
};