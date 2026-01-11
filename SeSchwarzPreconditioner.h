/////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2002 - 2022,
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "SeCsr.h"
#include "SeMorton.h"
#include "SeAabbSimd.h"
#include "SeCollisionElements.h"


SE_NAMESPACE_BEGIN

// 實現 GPU-based Multilevel Additive Schwarz (MAS) 預處理器的主要類別。
class SeSchwarzPreconditioner
{

public:
	
	//==== input data

	const SeVec3fSimd* m_positions = nullptr;		// the mesh nodal positions

	//==== input mesh topology data for collision computation

	const Int4* m_edges = nullptr;				// indices of the two adjacent vertices and the two opposite vertices of edges
	const Int4* m_faces = nullptr;				// indices of the three adjacent vertices of faces, the fourth is useless
	
	const SeCsr<int>* m_neighbours = nullptr;	// the adjacent information of vertices stored in a csr format

public:

	//==== call before time integration once a frame
	// [對應論文 Section 5: Multilevel Domain Construction]
	// 初始化預處理器，包含記憶體配置與初步的空間排序。
	void AllocatePrecoditioner(int numVerts, int numEdges, int numFaces);

	//==== call before PCG iteration loop
	// [對應論文 Section 6: Matrix Precomputation]
	// [對應論文 Algorithm 1: Matrix Precomputation]
	// 在每次 PCG 迭代前呼叫，負責組裝系統矩陣並計算子區域矩陣的逆 (Inversion)。
	// 包含 PrepareCollisionHessian, PrepareHessian, LDLtInverse512 等步驟。
	void PreparePreconditioner(const SeMatrix3f* diagonal, const SeMatrix3f* csrOffDiagonals, const int* csrRanges,
		const EfSet* efSets, const EeSet* eeSets, const VfSet* vfSets, unsigned int* efCounts, unsigned int* eeCounts, unsigned int* vfCounts);

	//==== call during PCG iterations
	// [對應論文 Section 7: Runtime Preconditioning]
	// [對應論文 Equation 5]
	// 執行預處理運算： z = M_{MAS}^{-1} * r
	// M_{MAS}^{-1} = M_{(0)}^{-1} + Sum( C_{(l)}^T * M_{(l)}^{-1} * C_{(l)} )
	void Preconditioning(SeVec3fSimd* z, const SeVec3fSimd* residual, int dim);

private:

	int m_numVerts = 0;
	int m_numEdges = 0;
	int m_numFaces = 0;

	int m_frameIndex = 0;

	// [對應論文 Section 6.1] "In this work, we set M = 32"
	// 每個子區域 (Domain) 的大小 (bank size)。
	int m_totalSz = 0;
	// [對應論文 Section 5.2] 層級總數 (L)。
	int m_numLevel = 0;

	int m_totalNumberClusters;

	
	SeAabb<SeVec3fSimd>					m_aabb; // 用於計算 Morton Code 的邊界框


	SeArray2D<SeMatrix3f>				m_hessianMapped;
	std::vector<int>					m_mappedNeighborsNum;
	std::vector<int>					m_mappedNeighborsNumRemain;
	SeArray2D<int>						m_mappedNeighbors;
	SeArray2D<int>						m_mappedNeighborsRemain;

	// [對應論文 Section 5.2]
	// 儲存粗糙空間層級關係的表格，對應論文中的限制算子 C_(l) 或映射 map_(l)。
	SeArray2D<int>						m_CoarseSpaceTables;

	// 以下是用於層級建構 (Algorithm 2) 的輔助資料結構
	std::vector<int>                    m_prefixOrignal;
	std::vector<unsigned int>           m_fineConnectMask;  // [對應 Section 5.2] 用於判斷連接性
	std::vector<unsigned int>           m_nextConnectMsk;
	std::vector<unsigned int>           m_nextPrefix;

	std::vector<SeVec3fSimd>			m_mappedPos;


	std::vector<Int4>					m_coarseTables; // 壓縮後的層級映射表
	std::vector<int>					m_goingNext; // 指向下一層級對應節點的索引
	std::vector<int>					m_denseLevel;
	std::vector<Int2>					m_levelSize;  // .x = current level size    .y = (prefixed) current level begin index

	// 預處理過程中的暫存向量
	std::vector<SeVec3fSimd>			m_mappedR; // [對應 Equation 5] 限制後的殘差 C_(l) * r
	std::vector<SeVec3fSimd>			m_mappedZ; // [對應 Equation 5] 局部求解後的結果 y_(l)
	std::vector<int>                    m_MapperSortedGetOriginal;          // sorted by morton
	std::vector<int>                    m_mapperOriginalGetSorted;
	std::vector<SeMorton64>             m_mortonCode;

	SeArray2D<SeMatrix3f>				m_hessian32;
	std::vector<SeMatrix3f>             m_additionalHessian32;
	std::vector<float>                  m_invSymR;

	
	int m_stencilNum;
	int m_maxStencilNum;
	std::vector<Stencil>				m_stencils;
	std::vector<Int5>					m_stencilIndexMapped;

private:

	void ComputeLevelNums(int bankSize);

	void DoAlllocation();

	void ComputeTotalAABB();

	void ComputeAABB();

	void SpaceSort();

	void FillSortingData();

	void DoingSort();

	void ComputeInverseMapper();

	void MapHessianTable();

	void PrepareCollisionStencils(const EfSet* efSets, const EeSet* eeSets, const VfSet* vfSets, unsigned int* efCounts, unsigned int* eeCounts, unsigned int* vfCounts);

	void MapCollisionStencilIndices();

	void ReorderRealtime();



	void BuildConnectMaskL0();

	void BuildCollisionConnection(unsigned int* pConnect, const int* pCoarseTable);

	void PreparePrefixSumL0();

	void BuildLevel1();

	void BuildConnectMaskLx(int level);

	void NextLevelCluster(int level);

	void PrefixSumLx(int level);

	void ComputeNextLevel(int level);

	void TotalNodes();

	void AggregationKernel();

	void AdditionalSchwarzHessian2(SeMatrix3f hessian, std::vector<SeMatrix3f>& pAdditionalHessian, SeArray2D<SeMatrix3f>& pDenseHessian, int v1, int v2, const std::vector<int>& pGoingNext, int nLevel, int vNum);

	void PrepareCollisionHessian();

	void PrepareHessian(const SeMatrix3f* diagonal, const SeMatrix3f* csrOffDiagonals, const int* csrRanges);
	
	void LDLtInverse512();

	void BuildResidualHierarchy(const SeVec3fSimd* m_cgResidual);

	void SchwarzLocalXSym();

	void CollectFinalZ(SeVec3fSimd* m_cgZ);
};


SE_NAMESPACE_END
