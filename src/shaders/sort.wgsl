const RADIX_BASE = 256u;
const RADIX_MASK = 255u;
const RADIX_BITS_PER_DIGIT = 8u;
const RADIX_DIGIT_PLACES = 4u;
const ENTRIES_PER_INVOCATION = 4u;
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

		var key: u32 = ~0x0u; // Stream compaction for frustum culling
		let pos = readVec3(splat_index + posOffset);
		var clipPos = cam.projection * cam.view * vec4f(pos, 1);
		clipPos /= clipPos.w;
		if (isInFrustum(clipPos.xyz)) {
			// key = bitcast<u32>(clipPos.z);
			key = bitcast<u32>(distance(cam.position, pos));
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

// Blelloch exclusive prefix sum algorithm
@compute @workgroup_size(RADIX_BASE)
fn exPrefixSum(
	@builtin(local_invocation_id) localID: vec3<u32>,
	@builtin(global_invocation_id) globalID: vec3<u32>
) {
	// Reduce
	var offset = 1u;
	for (var d = RADIX_BASE >> 1u; d > 0u; d >>= 1u) {
		workgroupBarrier();
		if(localID.x < d) {
			var leftIndex = offset * ((localID.x << 1) + 1u) - 1u;
			var rightIndex = offset * ((localID.x << 1) + 2u) - 1u;

			atomicAdd(&sorting.digitHistogram[globalID.y][rightIndex], atomicLoad(&sorting.digitHistogram[globalID.y][leftIndex]));
		}
		offset <<= 1u;
	}

	if (localID.x == 0u) {
		atomicStore(&sorting.digitHistogram[globalID.y][RADIX_BASE - 1], 0u);
	}

	// Downsweep
	for (var d = 1u; d < RADIX_BASE; d <<= 1u) {
		workgroupBarrier();
		offset >>= 1u;
		if (localID.x < d) {
			var leftIndex = offset * ((localID.x << 1) + 1u) - 1u;
			var rightIndex = offset * ((localID.x << 1) + 2u) - 1u;

			atomicStore(
				&sorting.digitHistogram[globalID.y][leftIndex],
				atomicAdd(
					&sorting.digitHistogram[globalID.y][rightIndex],
					atomicLoad(&sorting.digitHistogram[globalID.y][leftIndex])
				)
			);
		}
	}
	workgroupBarrier();
}

struct LocalBinning {
	entries: array<atomic<u32>, WORKGROUP_ENTRIES_BINNING>,
	scan: array<atomic<u32>, RADIX_BASE>,
}

var<workgroup> localBinning: LocalBinning;

// same as above, different elements array
fn exPrefixScan(index: u32) -> u32 {
	var offset = 1u;
	for (var d = RADIX_BASE >> 1; d > 0; d >>= 1) {
		workgroupBarrier();
		if(index < d) {
			var leftIndex = offset * (2 * index + 1) - 1;
			var rightIndex = offset * (2 * index + 2) - 1;

			atomicAdd(&localBinning.scan[rightIndex], atomicLoad(&localBinning.scan[leftIndex]));
		}
		offset <<= 1;
	}

	if (index == 0) {
		atomicStore(&localBinning.scan[RADIX_BASE - 1], 0);
	}

	for (var d = 1u; d < RADIX_BASE; d <<= 1) {
		workgroupBarrier();
		offset >>= 1;
		if (index < d) {
			var leftIndex = offset * (2 * index + 1) - 1;
			var rightIndex = offset * (2 * index + 2) - 1;

			atomicStore(
				&localBinning.scan[leftIndex],
				atomicAdd(
					&localBinning.scan[rightIndex],
					atomicLoad(&localBinning.scan[leftIndex])
				)
			);
		}
	}
	workgroupBarrier();
	return atomicLoad(&localBinning.scan[index]);
}

@compute @workgroup_size(RADIX_BASE)
fn bin(@builtin(local_invocation_id) localID: vec3<u32>) {
	// Draw an assignment number
	if (localID.x == 0u) {
		atomicStore(&localBinning.entries[0], atomicAdd(&sorting.assignmentCounter, 1u));
	}
	// Reset histogram
	atomicStore(&localBinning.scan[localID.x], 0u);
	workgroupBarrier();

	let assignment = atomicLoad(&localBinning.entries[0]);
	let global_entry_offset = assignment * WORKGROUP_ENTRIES_BINNING;
	// TODO: Specialize end shader
	if (localID.x == 0u && (global_entry_offset + WORKGROUP_ENTRIES_BINNING) >= arrayLength(&entriesIn)) {
		// Last workgroup resets the assignment number for the next pass
		atomicStore(&sorting.assignmentCounter, 0u);
	}

	// Load keys from global memory into registers and rank them
	var keys: array<u32, ENTRIES_PER_INVOCATION>;
	var ranks: array<u32, ENTRIES_PER_INVOCATION>;
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		keys[entry_index] = entriesIn[global_entry_offset + RADIX_BASE * entry_index + localID.x].key;
		let digit = (keys[entry_index] >> (passIndex * RADIX_BITS_PER_DIGIT)) & RADIX_MASK;

		// Everything goes wrong here and here alone.
		// For the sort to be stable, we need these rankings to be ascending within the same workgroup
		// i.e, if thread 0 and thread 1 have an entry with the same digit, the given rank in thread 0 should be lower than the one in thread 1,
		// but we have no guarantee that the atomicAdd will give this result.
		// TODO: investigate alternatives such as faking WLMS using workgroups and shared memory
		// https://github.com/googlefonts/compute-shader-101/blob/9f882d8d7d2fad98372d04350020c6cd672c1a72/compute-shader-hello/src/shader.wgsl#L371-L461
		ranks[entry_index] = atomicAdd(&localBinning.scan[digit], 1u);
	}
	workgroupBarrier();

	// Cumulate histogram
	let local_digit_count = atomicLoad(&localBinning.scan[localID.x]);
	let local_digit_offset = exPrefixScan(localID.x);

	// Chained scan decoupled lookback
	atomicStore(&statusCounter[assignment][localID.x], STATUS_LOCAL | local_digit_count);

	// inclusive scan value
	// "how many of this digit have blocks with lower assignment seen"
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
		ranks[entry_index] += atomicLoad(&localBinning.scan[digit]);
		atomicStore(&localBinning.entries[ranks[entry_index]], key);
	}
	workgroupBarrier();

	// Add global offset
	atomicStore(&localBinning.scan[localID.x], global_digit_count - local_digit_offset);
	workgroupBarrier();

	// Store keys from shared memory into global memory
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let key = atomicLoad(&localBinning.entries[RADIX_BASE * entry_index + localID.x]);
		let digit = (key >> (passIndex * RADIX_BITS_PER_DIGIT)) & RADIX_MASK;
		keys[entry_index] = digit;
		entriesOut[atomicLoad(&localBinning.scan[digit]) + RADIX_BASE * entry_index + localID.x].key = key;
	}
	workgroupBarrier();

	// Load values from global memory and scatter them inside shared memory
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let value = entriesIn[global_entry_offset + RADIX_BASE * entry_index + localID.x].value;
		atomicStore(&localBinning.entries[ranks[entry_index]], value);
	}
	workgroupBarrier();

	// Store values from shared memory into global memory
	for (var entry_index = 0u; entry_index < ENTRIES_PER_INVOCATION; entry_index += 1u) {
		let value = atomicLoad(&localBinning.entries[RADIX_BASE * entry_index + localID.x]);
		let digit = keys[entry_index];
		entriesOut[atomicLoad(&localBinning.scan[digit]) + RADIX_BASE * entry_index + localID.x].value = value;
	}
}