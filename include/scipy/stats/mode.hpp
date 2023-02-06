/*
Scientific methods on top of NP library

Copyright (c) 2023 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <queue>
#include <unordered_map>
#include <utility>

#include <np/Array.hpp>
#include <pd/core/frame/DataFrame/DataFrame.hpp>

namespace scipy {
    namespace stats {
        template<typename DType>
        using Array = np::Array<DType>;
        // Return an array of the modal (most common) value in the passed array.
        //
        // If there is more than one such value, only one is returned. The bin-count for
        // the modal bins is also returned.
        template<typename DType = np::DTypeDefault>
        std::pair<Array<DType>, Array<np::Size>> mode(const Array<DType> &array) {
            auto sh = array.shape();
            if (sh.empty()) {
                Array<DType> mode{};
                Array<np::Size> count{};
                return std::make_pair(mode, count);
            }
            if (sh.size() != 1 && sh.size() != 2)
                throw std::runtime_error("Only 1D and 2D arrays supported");

            using Pair = std::pair<DType, np::Size>;
            auto cmp = [](const Pair &pair1, const Pair &pair2) {
                return pair1.second < pair2.second || (pair1.second == pair2.second && pair1.first > pair2.first);
            };
            if (sh.size() == 1) {
                std::unordered_map<DType, np::Size> freq;
                for (auto it = array.cbegin(); it != array.cend(); ++it) {
                    ++freq[*it];
                }
                std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> queue{cmp};
                for (auto it = freq.cbegin(); it != freq.cend(); ++it) {
                    queue.push(*it);
                }
                DType maxMode{queue.top().first};
                np::Size maxCount{queue.top().second};
                Array<DType> mode{{maxMode}, np::Shape{1}};
                Array<np::Size> count{{maxCount}, np::Shape{1}};
                return std::make_pair(mode, count);
            }
            std::vector<DType> maxMode;
            maxMode.resize(sh[1]);
            std::vector<np::Size> maxCount;
            maxCount.resize(sh[1]);
            for (np::Size dim = 0; dim < sh[1]; ++dim) {
                std::unordered_map<DType, np::Size> freq;
                for (std::size_t index = 0; index < sh[0]; ++index) {
                    ++freq[array.get(index * sh[1] + dim)];
                }
                std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> queue{cmp};
                for (const auto &it: freq) {
                    queue.emplace(it);
                }
                maxMode[dim] = queue.top().first;
                maxCount[dim] = queue.top().second;
            }
            Array<DType> mode{maxMode, np::Shape{1, sh[1]}};
            Array<np::Size> count{maxCount, np::Shape{1, sh[1]}};
            return std::make_pair(mode, count);
        }

        using DataFrame = pd::DataFrame;
        std::pair<DataFrame, DataFrame> mode(const DataFrame &dataFrame) {
            auto sh = dataFrame.shape();
            if (sh.empty()) {
                DataFrame mode{};
                DataFrame count{};
                return std::make_pair(mode, count);
            }
            if (sh.size() != 1 && sh.size() != 2)
                throw std::runtime_error("Only 1D and 2D arrays supported");

            using Pair = std::pair<pd::internal::Value, np::Size>;
            auto cmp = [](const Pair &pair1, const Pair &pair2) {
                return pair1.second < pair2.second || (pair1.second == pair2.second && pair1.first > pair2.first);
            };
            if (sh.size() == 1) {
                std::unordered_map<pd::internal::Value, np::Size> freq;
                for (np::Size row = 0; row < dataFrame.shape()[0]; ++row) {
                    ++freq[dataFrame.at(row, 0)];
                }
                std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> queue{cmp};
                for (const auto &it: freq) {
                    queue.emplace(it);
                }
                pd::internal::Value maxMode{queue.top().first};
                np::Size maxCount{queue.top().second};
                DataFrame mode{np::Array<pd::internal::Value>{std::vector<pd::internal::Value>{maxMode}}};
                DataFrame count{np::Array<np::Size>{std::vector<np::Size>{maxCount}}};
                return std::make_pair(mode, count);
            }
            std::vector<pd::internal::Value> maxMode;
            maxMode.resize(sh[1]);
            std::vector<np::Size> maxCount;
            maxCount.resize(sh[1]);
            for (np::Size dim = 0; dim < sh[1]; ++dim) {
                std::unordered_map<pd::internal::Value, np::Size> freq;
                for (std::size_t index = 0; index < sh[0]; ++index) {
                    ++freq[dataFrame.at(index, dim)];
                }
                std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> queue{cmp};
                for (const auto &it: freq) {
                    queue.emplace(it);
                }
                maxMode[dim] = queue.top().first;
                maxCount[dim] = queue.top().second;
            }
            DataFrame mode{Array<pd::internal::Value>{maxMode, np::Shape{1, sh[1]}}};
            DataFrame count{Array<np::Size>{maxCount, np::Shape{1, sh[1]}}};
            return std::make_pair(mode, count);
        }
    }// namespace stats
}// namespace scipy