const RADIX_BITS_PER_DIGIT = 8u;
const RADIX_DIGIT_PLACES = 4u;
const ENTRIES_PER_INVOCATION = 4u;
const RADIX_BASE = 1u << RADIX_BITS_PER_DIGIT;
const RADIX_MASK = RADIX_BASE - 1;
const WORKGROUP_ENTRIES_BINNING = RADIX_BASE * ENTRIES_PER_INVOCATION;

const STATUS_NOT_READY = 0x00000000u;
const STATUS_LOCAL = 0x40000000u;
const STATUS_GLOBAL = 0x80000000u;
const STATUS_MASK = 0xC0000000u;
const VALUE_MASK = ~STATUS_MASK;

struct DrawIndirect {
	vertexCount: u32,
	instanceCount: atomic<u32>,
	_baseVertex: u32,
	_baseInstance: u32,
};

struct SortingGlobal {
	digitHistogram: array<array<atomic<u32>, RADIX_BASE>, RADIX_DIGIT_PLACES>,
	assignmentCounter: atomic<u32>,
};

@group(0) @binding(2) var<uniform> passIndex: u32;
@group(0) @binding(3) var<storage, read_write> sorting: SortingGlobal;
@group(0) @binding(4) var<storage, read_write> statusCounter: array<array<atomic<u32>, RADIX_BASE>, MAX_TILE_COUNT_BINNING>;
@group(0) @binding(5) var<storage, read_write> drawIndirect: DrawIndirect;
@group(0) @binding(6) var<storage, read_write> entriesIn: array<Entry>;
@group(0) @binding(7) var<storage, read_write> entriesOut: array<Entry>;

// Onesweep Radix Sort
// Based on https://github.com/Lichtso/splatter
// We use a slower but exact key ranking algorithm
var<workgroup> localHistogram: array<array<atomic<u32>, RADIX_BASE>, RADIX_DIGIT_PLACES>;

@compute @workgroup_size(RADIX_BASE, RADIX_DIGIT_PLACES)
fn histogram(
	@builtin(local_invocation_id) localID: vec3<u32>,
	@builtin(global_invocation_id) globalID: vec3<u32>,
) {
	let thread_index = globalID.x * RADIX_DIGIT_PLACES + globalID.y;
	let start_entry_index = thread_index * ENTRIES_PER_INVOCATION;
	let end_entry_index = start_entry_index + ENTRIES_PER_INVOCATION;
	for (var entry_index = start_entry_index; entry_index < end_entry_index; entry_index += 1u) {
		let splat_index = entry_index * stride;
		if (splat_index >= arrayLength(&gaussianData)) { continue; }

		var key: u32 = ~0u; // Stream compaction for frustum culling
		let pos = readVec3(splat_index + posOffset);
		var clipPos = cam.projection * cam.view * vec4f(pos, 1);
		clipPos /= clipPos.w;
		if (isInFrustum(clipPos.xyz)) {
			key = bitcast<u32>(1.0 - clipPos.z);
		}

		entriesOut[entry_index] = Entry(key, entry_index);
		for (var shift = 0u; shift < RADIX_DIGIT_PLACES; shift += 1u) {
			let digit = (key >> (shift * RADIX_BITS_PER_DIGIT)) & RADIX_MASK;
			atomicAdd(&localHistogram[shift][digit], 1u);
		}
	}
	workgroupBarrier();

	atomicAdd(&sorting.digitHistogram[localID.y][localID.x], atomicLoad(&localHistogram[localID.y][localID.x]));
}

@compute @workgroup_size(1)
fn exPrefixSum(@builtin(global_invocation_id) globalID: vec3<u32>) {
    var sum = 0u;
	for (var digit = 0u; digit < RADIX_BASE; digit += 1u) {
		sum += atomicExchange(&sorting.digitHistogram[globalID.y][digit], sum);
	}
}

const WARPSIZE = 32u;
const WARPCOUNT = RADIX_BASE / WARPSIZE;

var<workgroup> entries: array<atomic<u32>, WORKGROUP_ENTRIES_BINNING>;
var<workgroup> scan: array<atomic<u32>, RADIX_BASE>;
var<workgroup> warpHists: array<array<u32, RADIX_BASE>, WARPCOUNT>;
var<workgroup> warpKeys: array<array<u32, WARPSIZE>, WARPCOUNT>;

// Blelloch exclusive prefix sum algorithm
fn exclusiveScan(index: u32) -> u32 {
	var offset = 1u;
	for (var d = RADIX_BASE >> 1; d > 0; d >>= 1) {
		workgroupBarrier();
		if(index < d) {
			var leftIndex = offset * (2 * index + 1) - 1;
			var rightIndex = offset * (2 * index + 2) - 1;

			atomicAdd(&scan[rightIndex], atomicLoad(&scan[leftIndex]));
		}
		offset <<= 1;
	}

	if (index == 0) {
		atomicStore(&scan[RADIX_BASE - 1], 0);
	}

	for (var d = 1u; d < RADIX_BASE; d <<= 1) {
		workgroupBarrier();
		offset >>= 1;
		if (index < d) {
			var leftIndex = offset * (2 * index + 1) - 1;
			var rightIndex = offset * (2 * index + 2) - 1;

			atomicStore(
				&scan[leftIndex],
				atomicAdd(
					&scan[rightIndex],
					atomicLoad(&scan[leftIndex])
				)
			);
		}
	}
	workgroupBarrier();
	return atomicLoad(&scan[index]);
}

@compute @workgroup_size(RADIX_BASE)
fn bin(@builtin(local_invocation_id) localID: vec3<u32>) {
	// Draw an assignment number
	if (localID.x == 0u) {
		atomicStore(&entries[0], atomicAdd(&sorting.assignmentCounter, 1u));
	}
	// Reset histogram
	atomicStore(&scan[localID.x], 0u);
	workgroupBarrier();

	let assignment = atomicLoad(&entries[0]);
	let global_entry_offset = assignment * WORKGROUP_ENTRIES_BINNING;
	if (localID.x == 0u && (global_entry_offset + WORKGROUP_ENTRIES_BINNING) >= arrayLength(&entriesIn)) {
		// Last workgroup resets the assignment number for the next pass
		atomicStore(&sorting.assignmentCounter, 0u);
	}

	// Load keys from global memory into registers and rank them
	// Shared memory implementation of warp-level multisplit
	// taken from https://github.com/googlefonts/compute-shader-101/blob/9f882d8d7d2fad98372d04350020c6cd672c1a72/compute-shader-hello/src/shader.wgsl#L371-L461
	let warpIdx = localID.x / WARPSIZE;
	let laneIdx = localID.x % WARPSIZE;
	var keys: array<u32, ENTRIES_PER_INVOCATION>;
	var ranks: array<u32, ENTRIES_PER_INVOCATION>;
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {

		let idx = global_entry_offset + RADIX_BASE * entry_index + localID.x;
		var key = ~0u;
		if (idx < arrayLength(&entriesIn)) {
			key = entriesIn[idx].key;
		}

		keys[entry_index] = key;
		let digit = (key >> (passIndex * RADIX_BITS_PER_DIGIT)) & RADIX_MASK;

		warpKeys[warpIdx][laneIdx] = digit;
		workgroupBarrier();

		var ballot = 0u;
		for (var j = 0u; j < WARPSIZE; j++) {
			if (digit == warpKeys[warpIdx][j]) {
				ballot |= (1u << j);
			}
		}

		let rank = countOneBits(ballot << (31u - laneIdx)) - 1;
		ranks[entry_index] = warpHists[warpIdx][digit] + rank;

		if (rank == 0u) {
			warpHists[warpIdx][digit] += countOneBits(ballot);
		}
	}
	workgroupBarrier();

	var local_digit_count = 0u;
	for (var i = 0u; i < WARPCOUNT; i += 1u) {
		let temp = warpHists[i][localID.x];
		warpHists[i][localID.x] = local_digit_count;
		local_digit_count += temp;
	}
	
	atomicStore(&scan[localID.x], local_digit_count);
	let local_digit_offset = exclusiveScan(localID.x);

	// Chained scan decoupled lookback
	atomicStore(&statusCounter[assignment][localID.x], STATUS_LOCAL | local_digit_count);
	var global_digit_count = 0u;
	var previous_tile = assignment;
	while true {
		if (previous_tile == 0u) {
			global_digit_count += atomicLoad(&sorting.digitHistogram[passIndex][localID.x]);
			break;
		}

		previous_tile -= 1u;
		var status_counter = STATUS_NOT_READY;
		while ((status_counter & STATUS_MASK) == STATUS_NOT_READY) {
			status_counter = atomicLoad(&statusCounter[previous_tile][localID.x]);
		}
		
		global_digit_count += status_counter & VALUE_MASK;
		if ((status_counter & STATUS_GLOBAL) != 0u) {
			break;
		}
	}

	atomicStore(&statusCounter[assignment][localID.x], STATUS_GLOBAL | (global_digit_count + local_digit_count));
	if (passIndex == RADIX_DIGIT_PLACES - 1u && localID.x == RADIX_BASE - 2u && (global_entry_offset + WORKGROUP_ENTRIES_BINNING) >= arrayLength(&entriesIn)) {
		drawIndirect.vertexCount = 4u;
		atomicStore(&drawIndirect.instanceCount, global_digit_count + local_digit_count);
	}

	// Scatter keys inside shared memory
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let key = keys[entry_index];
		let digit = (key >> (passIndex * RADIX_BITS_PER_DIGIT)) & RADIX_MASK;
		ranks[entry_index] += warpHists[warpIdx][digit] + atomicLoad(&scan[digit]);
		atomicStore(&entries[ranks[entry_index]], key);
	}
	workgroupBarrier();

	// Add global offset
	atomicStore(&scan[localID.x], global_digit_count - local_digit_offset);
	workgroupBarrier();

	// Store keys from shared memory into global memory
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let key = atomicLoad(&entries[RADIX_BASE * entry_index + localID.x]);
		let digit = (key >> (passIndex * RADIX_BITS_PER_DIGIT)) & RADIX_MASK;
		keys[entry_index] = digit;
		entriesOut[atomicLoad(&scan[digit]) + RADIX_BASE * entry_index + localID.x].key = key;
	}
	workgroupBarrier();

	// Load values from global memory and scatter them inside shared memory
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let value = entriesIn[global_entry_offset + RADIX_BASE * entry_index + localID.x].value;
		atomicStore(&entries[ranks[entry_index]], value);
	}
	workgroupBarrier();

	// Store values from shared memory into global memory
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let value = atomicLoad(&entries[RADIX_BASE * entry_index + localID.x]);
		let digit = keys[entry_index];
		entriesOut[atomicLoad(&scan[digit]) + RADIX_BASE * entry_index + localID.x].value = value;
	}
}