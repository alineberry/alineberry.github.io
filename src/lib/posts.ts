import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

export function readingTime(body: string): number {
  // Strip math blocks ($$...$$ and $...$), HTML tags, code fences,
  // and image markdown so we don't count LaTeX tokens as words.
  const cleaned = body
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/(?<!\\)\$[^$\n]+\$/g, ' ')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`[^`\n]+`/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[[^\]]*\]\([^)]+\)/g, ' ');
  const words = cleaned.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 200));
}

export function postHref(p: Post): string {
  // Normalize permalink to /foo/ (trailing slash, no leading dup)
  const raw = p.data.permalink.replace(/^\/+/, '').replace(/\/+$/, '');
  return '/' + raw + '/';
}

export async function getAllPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', (p) => !p.data.draft);
  return posts.sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );
}

export async function getEssays(): Promise<Post[]> {
  return (await getAllPosts()).filter((p) => !p.data.isNote);
}

export async function getNotes(): Promise<Post[]> {
  return (await getAllPosts()).filter((p) => p.data.isNote);
}
