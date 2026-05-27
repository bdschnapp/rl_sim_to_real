#!/usr/bin/env bash
set -e

: "${COLCON_PARALLEL_WORKERS:=1}"
: "${CMAKE_BUILD_PARALLEL_LEVEL:=2}"
export CMAKE_BUILD_PARALLEL_LEVEL

sanitize_missing_local_setup_sources() {
  local package_dsv
  local prefix
  local tmp
  local changed
  local line
  local rel_path
  local source_path

  shopt -s nullglob
  for package_dsv in install/*/share/*/package.dsv; do
    prefix="${package_dsv%%/share/*}"
    tmp="${package_dsv}.tmp"
    changed=0
    : > "${tmp}"

    while IFS= read -r line || [ -n "${line}" ]; do
      case "${line}" in
        source\;*local_setup.*)
          rel_path="${line#source;}"
          source_path="${prefix}/${rel_path}"
          if [ ! -e "${source_path}" ]; then
            changed=1
            continue
          fi
          ;;
      esac
      printf '%s\n' "${line}" >> "${tmp}"
    done < "${package_dsv}"

    if [ "${changed}" -eq 1 ]; then
      mv "${tmp}" "${package_dsv}"
    else
      rm "${tmp}"
    fi
  done
  shopt -u nullglob
}

: "${EXTRA_CXX_FLAGS:=-Wno-maybe-uninitialized -Wno-array-bounds}"

colcon build \
  --symlink-install \
  --parallel-workers "${COLCON_PARALLEL_WORKERS}" \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}"

sanitize_missing_local_setup_sources

echo "Build complete. Run 'source install/setup.bash' in your shell to use this workspace."
