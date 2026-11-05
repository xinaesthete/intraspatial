// "Open folder… / Open .zip…" beside a DOM page's store-URL input. The React pages use
// `LocalStorePicker.tsx`; both register the pick through `datasource/localStores`.

import { openLocalFolder, openLocalZip } from "./datasource/localStores";

/**
 * Put the picker after `input`. A pick writes its `local:` key into the input and fires `change`
 * (which the pages that reopen on a new URL already listen for), then calls `onPicked` for pages
 * that load from a button instead.
 */
export function attachLocalStorePicker(input: HTMLInputElement, onPicked?: (key: string) => void): void {
  const box = document.createElement("span");
  box.className = "local-store-picker";
  box.style.cssText = "display:inline-flex;gap:4px;align-items:center;flex-wrap:wrap";
  const status = document.createElement("span");
  status.style.cssText = "font-size:11px;opacity:0.75";

  const picked = (open: () => Promise<string>, what: string) => {
    status.textContent = `opening ${what}…`;
    open().then(
      (key) => {
        status.textContent = "";
        input.value = key;
        input.dispatchEvent(new Event("change"));
        onPicked?.(key);
      },
      (err: unknown) => {
        status.textContent = `could not open ${what}: ${err instanceof Error ? err.message : String(err)}`;
      },
    );
  };

  const button = (label: string, title: string, setup: (file: HTMLInputElement) => void) => {
    const file = document.createElement("input");
    file.type = "file";
    file.hidden = true;
    setup(file);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.title = title;
    btn.addEventListener("click", () => file.click());
    box.append(btn, file);
    return file;
  };

  const folder = button("Open folder…", "Pick a .zarr folder on this computer", (f) => f.setAttribute("webkitdirectory", ""));
  folder.addEventListener("change", () => {
    // Copied out first: clearing `value` empties the input's FileList in place.
    const files = [...(folder.files ?? [])];
    folder.value = ""; // so picking the same folder again still fires `change`
    if (files.length) picked(() => openLocalFolder(files), "folder");
  });
  const zip = button("Open .zip…", "Pick a zipped .zarr store on this computer", (f) => (f.accept = ".zip"));
  zip.addEventListener("change", () => {
    const file = zip.files?.[0];
    zip.value = "";
    if (file) picked(() => openLocalZip(file), file.name);
  });

  box.append(status);
  input.after(box);
}
